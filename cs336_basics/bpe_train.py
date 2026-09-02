from typing import Dict, BinaryIO
import regex as re
from pretokenization_example import find_chunk_boundaries


# assign initial token IDs to all possible 1-byte values
token_to_bytes: Dict[int, bytes] = {}
bytes_to_token = Dict[bytes, int] = {}
for i in range(256):
    token_to_bytes[i] = bytes([i])
    bytes_to_token[bytes([i])] = i

# Add delimiter as special token
DELIMITER = b'<|endoftext|>'
token_to_bytes[len(token_to_bytes)] = DELIMITER

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def read_until_delimiter(f: BinaryIO, delimiter: bytes) -> bytes:
    CHUNK_SIZE = 1 * 1024 # 1 KiB
    tmp_string: bytearray = []
    while True:
        start_cursor = f.tell()
        chunk = f.read(CHUNK_SIZE)

        if chunk == b'':
            return bytes(tmp_string)
        
        found_at = chunk.find(delimiter)
        if found_at != -1:
            tmp_string.extend(chunk[:found_at])
            f.seek(start_cursor + found_at + len(delimiter))
            return bytes(tmp_string)
        
        tmp_string.extend(chunk)


# Open training data in split into chunks
chunk_boundaries: list[int]
with open('../data/TinyStoriesV2-GPT4-valid.txt', 'rb') as f:
    chunk_boundaries = find_chunk_boundaries(f, 8, DELIMITER)

frequency_table: dict[tuple[bytes, ...], int] = {}
with open('../data/TinyStoriesV2-GPT4-valid.txt', 'rb') as f:
    while True:
        # Read until we find a delimiter
        output = read_until_delimiter(f, DELIMITER)

        if output == b'':
            break

        # Process the portion before the delimiter
        for group in re.findall(PAT, output.decode('utf-8')):
            group_byte = tuple(group.encode('utf-8'))
            if group_byte in frequency_table:
                frequency_table[group_byte] += 1
            else:
                frequency_table[group_byte] = 1

print(frequency_table)
