from typing import Dict, BinaryIO
import regex as re
import os

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def cam_generate_frequency_map(
    input_path: str | os.PathLike,
    start_index: int,
    end_index: int,
    delimiters: list[str]
) -> dict[tuple[bytes, ...], int]:
    frequency_table: dict[tuple[bytes, ...], int] = {}
    with open(input_path, 'rb') as f:
        f.seek(start_index)
        read_data = f.read(end_index - start_index)
        text_data = read_data.decode('utf-8')

        segments = re.split("|".join(map(re.escape, delimiters)), text_data)
        for seg in segments:
            for group in re.finditer(PAT, seg):
                group_tup = tuple(bytes([b]) for b in group.group().encode('utf-8'))
                if group_tup in frequency_table:
                    frequency_table[group_tup] += 1
                else:
                    frequency_table[group_tup] = 1
    
    return frequency_table

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: list[str],
) -> list[int]:
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    # Convert special tokens to bytes and build pattern for binary matching
    special_tokens_bytes = [token.encode('utf-8') for token in split_special_tokens]
    pattern = b"|".join(map(re.escape, special_tokens_bytes))

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Search for special tokens directly in binary data
            match = re.search(pattern, mini_chunk)
            if match:
                chunk_boundaries[bi] = initial_position + match.end()
                break

            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


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
        chunk_boundaries = find_chunk_boundaries(f, 2, special_tokens)

    # Process each chunk into a frequency map in parallel
    frequency_maps: list[dict[tuple[bytes, ...], int]] = []
    for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
        frequency_maps.append(cam_generate_frequency_map(input_path, start, end, special_tokens))

    # Merge all frequency maps into one
    final_freq_map: dict[tuple[bytes, ...], int] = {}
    for freq_map in frequency_maps:
        for key, value in freq_map.items():
            if key in final_freq_map:
                final_freq_map[key] += value
            else:
                final_freq_map[key] = value

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
