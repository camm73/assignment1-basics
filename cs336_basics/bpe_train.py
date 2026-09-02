from typing import Dict
import regex as re
from torch import special
from cs336_basics.pretokenization_example import find_chunk_boundaries
import os

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def cam_generate_frequency_map(
    input_path: str | os.PathLike,
    start_index: int,
    end_index: int,
    delimiters: list[str]
) -> dict[tuple[bytes, ...], int]:
    frequency_table: dict[tuple[bytes, ...], int] = {}
    with open(input_path, 'r') as f:
        f.seek(start_index)

        # NOTE: Will need to chunk this further for larger files to avoid excessive memory usage
        read_data = f.read(end_index - start_index)
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

    # First we'll determine the pretokenization splits
    chunk_boundaries: list[int]
    with open(input_path, 'rb') as f:
        # TODO: Currently just using first special token, but this should be changed to use all
        chunk_boundaries = find_chunk_boundaries(f, 8, special_tokens[0].encode('utf-8'))

    # Process each chunk into a frequency map in parallel
    frequency_maps: list[dict[tuple[bytes, ...], int]] = []
    for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
        frequency_maps.append(cam_generate_frequency_map(input_path, start, end, special_tokens))
    
    # Merge all frequency maps into one
    final_freq_map: dict[tuple[bytes, ...], int] = {}
    for freq_map in frequency_maps:
        for key, value in freq_map.items():
            if key in final_freq_map:
                final_freq_map[key] = final_freq_map[key] + value
            else:
                final_freq_map[key] = value

    # init for BPE process
    # assign initial token IDs to all possible 1-byte values
    token_to_bytes: Dict[int, bytes] = {}
    for i in range(256):
        token_to_bytes[i] = bytes([i])

    # Treat delimiters specially up front
    for delimiter in special_tokens:
        token_to_bytes[len(token_to_bytes)] = delimiter

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
        top_pair = None
        top_count = 0
        for pair, count in merge_counter.items():
            if count > top_count:
                top_count = count
                top_pair = pair

        if top_pair is None:
            print("Nothing left to merge")
            break

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
    return token_to_bytes, merge_list
