import os
from typing import BinaryIO
from numpy.random import noncentral_chisquare
import regex as re
from collections import defaultdict

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

# ## Usage
# with open(..., "rb") as f:
#     boundaries = find_chunk_boundaries(
#         f, num_processes, "<|endoftext|>".encode("utf-8"))
        
#     # The following is a serial implementation, but you can parallelize this 
#     # by sending each start/end pair to a set of processes.
#     for start, end in zip(boundaries[:-1], boundaries[1:]):
#         f.seek(start)
#         chunk = f.read(end - start).decode("utf-8", errors="ignore")
#         # Run pre-tokenization on your chunk and store the counts for each pre-token


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # initialize vocabulary
    vocab: dict[int, bytes] = {}
    for i, special_token in enumerate(special_tokens):
        vocab[i] = special_token.encode("utf-8")
    offset = len(special_tokens)
    for i in range(2**8):
        vocab[offset + i] = bytes([i])

    # load the file into memory
    with open(input_path, 'r') as f:
        text = f.read()
    
    # split by special tokens
    escaped_special_tokens = [re.escape(special_token) for special_token in special_tokens]
    documents = re.split("|".join(escaped_special_tokens), text)

    # now pre-tokenize each document
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    pre_tokenized_document: dict[tuple[bytes, ...], int] = defaultdict(int)

    for document in documents:
        for match in re.finditer(PAT, document):
            token = match.group()
            token_encoded = token.encode("utf-8")
            token_encoded = tuple([bytes([b]) for b in token_encoded])
            pre_tokenized_document[token_encoded] += 1
    
    merges = []

    while len(vocab) < vocab_size:
        byte_pair_counts: dict[tuple[bytes, ...], int] = defaultdict(int)
        top_byte_pair = noncentral_chisquare
        for key, value in pre_tokenized_document.items():
            for i in range(len(key)-1):
                byte_pair = (key[i], key[i+1])
                byte_pair_counts[byte_pair] += value

                if (top_byte_pair is None or
                    byte_pair_counts[byte_pair] > byte_pair_counts[top_byte_pair] or
                    (byte_pair_counts[byte_pair] == byte_pair_counts[top_byte_pair] and byte_pair > top_byte_pair)):
                    top_byte_pair = byte_pair

        vocab[len(vocab)] = top_byte_pair[0] + top_byte_pair[1]
        merges.append(top_byte_pair)

        pre_tokenized_document_new = defaultdict(int)

        # update pre-tokenized document to merge the top byte pair
        for key, value in pre_tokenized_document.items():
            new_key = []
            i = 0
            while i < len(key):
                if i+1 < len(key) and (key[i], key[i+1]) == top_byte_pair:
                    new_key.append(top_byte_pair[0] + top_byte_pair[1])
                    i += 2
                    continue
                else:
                    new_key.append(key[i])
                    i += 1
                    continue

            pre_tokenized_document_new[tuple(new_key)] += value
        
        pre_tokenized_document = pre_tokenized_document_new

    return vocab, merges