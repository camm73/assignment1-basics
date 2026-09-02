from typing import Dict, BinaryIO
import regex as re
from pretokenization_example import find_chunk_boundaries


# assign initial token IDs to all possible 1-byte values
token_to_bytes: Dict[int, bytes] = {}
for i in range(256):
    token_to_bytes[i] = bytes([i])

# Add delimiter as special token
DELIMITER = b'<|endoftext|>'
token_to_bytes[len(token_to_bytes)] = DELIMITER

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Open training data in split into chunks
chunk_boundaries: list[int]
with open('../data/TinyStoriesV2-GPT4-valid.txt', 'rb') as f:
    chunk_boundaries = find_chunk_boundaries(f, 8, DELIMITER)

# Construct frequency table for the file
frequency_table: dict[tuple[bytes, ...], int] = {}
delimiter_list: list[str] = [DELIMITER.decode()]
with open('../data/TinyStoriesV2-GPT4-valid.txt', 'r') as f:
    read_data = f.read()
    segments = re.split("|".join(map(re.escape, delimiter_list)), read_data)
    for seg in segments:
        for group in re.finditer(PAT, seg):
            group_tup = tuple(bytes([b]) for b in group.group().encode('utf-8'))
            if group_tup in frequency_table:
                frequency_table[group_tup] += 1
            else:
                frequency_table[group_tup] = 1

start_frequency_table_size = len(frequency_table)
VOCAB_SIZE = 1000
while len(token_to_bytes) < VOCAB_SIZE:
    merge_counter: dict[tuple[bytes, ...], int] = {}

    assert len(frequency_table) == start_frequency_table_size

    # Count byte pair occurences and multiple by frequency in freq table
    for key, value in frequency_table.items():
        for i in range(len(key) - 1):
            one = key[i]
            two = key[i+1]
            tup = (one, two)
            assert type(one) == bytes
            assert type(two) == bytes

            if tup in merge_counter:
                merge_counter[tup] += 1 * value
            else:
                merge_counter[tup] = 1 * value
    
    # Find maximum byte pair occurence in merge_counter
    top_pair = None
    top_count = 0
    for pair, count in merge_counter.items():
        if count > top_count:
            top_count = count
            top_pair = pair

    if top_pair is None:
        print("Nothing left to merge")
        break

    # Replace instances of byte pair in frequency_table with new merged pair
    new_frequency_table: dict[tuple[bytes, ...], int] = {}
    for key, value in frequency_table.items():
        new_key: list[bytes] = []
        i = 0
        while i < len(key):
            one = key[i]
            # Need to handle case where we only have one character left
            if i == len(key) - 1:
                new_key.extend([one])
                break

            two = key[i+1]
            tup = (one, two)

            # if we find a pair, merge the bytes
            if tup == top_pair:
                new_key.extend([one + two])
                i += 2
            else:
                new_key.extend([one])
                i += 1

        # we'll keep the same value here, just replacing with the new key
        new_frequency_table[tuple(new_key)] = value

    # once done, we'll replace the old frequency map
    frequency_table = new_frequency_table

    # Update token_to_bytes mapping with new token ID and merged bytes
    print(f'Merging {top_pair[0]} and {top_pair[1]}')
    token_to_bytes[len(token_to_bytes)] = top_pair[0] + top_pair[1]
