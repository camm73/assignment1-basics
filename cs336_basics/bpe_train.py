from typing import Dict
import regex as re
from torch import special
from cs336_basics.pretokenization_example import find_chunk_boundaries
import os

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def cam_generate_frequency_map(
    input_path: str | os.PathLike,
    delimiters: list[str]
) -> dict[tuple[bytes, ...], int]:
    frequency_table: dict[tuple[bytes, ...], int] = {}
    with open(input_path, 'r') as f:
        # NOTE: Will need to chunk this further for larger files to avoid excessive memory usage
        read_data = f.read()
        segments = re.split("|".join(map(re.escape, delimiters)), read_data)
        for seg in segments:
            for group in re.finditer(PAT, seg):
                group_tup = tuple(bytes([b]) for b in group.group().encode('utf-8'))
                if group_tup in frequency_table:
                    frequency_table[group_tup] += 1
                else:
                    frequency_table[group_tup] = 1
    
    return frequency_table


def cam_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,):
    assert len(special_tokens) > 0
    final_freq_map = cam_generate_frequency_map(input_path, special_tokens)

    # init for BPE process
    # assign initial token IDs to all possible 1-byte values
    token_to_bytes: Dict[int, bytes] = {}
    for i in range(256):
        token_to_bytes[i] = bytes([i])

    # Treat delimiters specially up front
    for delimiter in special_tokens:
        token_to_bytes[len(token_to_bytes)] = delimiter.encode('utf-8')

    # Start BPE process
    start_frequency_table_size = len(final_freq_map)
    merge_list: list[tuple[bytes, bytes]] = []
    while len(token_to_bytes) < vocab_size:
        merge_counter: dict[tuple[bytes, ...], int] = {}

        assert len(final_freq_map) == start_frequency_table_size

        # Count byte pair occurences and multiple by frequency in freq table
        for key, value in final_freq_map.items():
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
        pair_candidates: list[tuple[bytes, ...]] = []
        top_count = 0
        for pair, count in merge_counter.items():
            if count > top_count:
                top_count = count
                pair_candidates = [pair]
            elif count == top_count:
                pair_candidates.append(pair)

        if len(pair_candidates) == 0:
            print("Nothing left to merge")
            break

        # Determine top_pair if there are ties as top lexicographical pair
        pair_candidates.sort()
        top_pair = pair_candidates[-1]

        # Add to merge list
        merge_list.append(top_pair)

        # Replace instances of byte pair in final_freq_map with new merged pair
        new_frequency_table: dict[tuple[bytes, ...], int] = {}
        for key, value in final_freq_map.items():
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
        final_freq_map = new_frequency_table

        # Update token_to_bytes mapping with new token ID and merged bytes
        token_to_bytes[len(token_to_bytes)] = top_pair[0] + top_pair[1]
    
    # Verify
    for key, value in token_to_bytes.items():
        assert type(key) == int
        assert type(value) == bytes

    return token_to_bytes, merge_list
