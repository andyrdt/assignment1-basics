import os
from typing import BinaryIO
from typing_extensions import Set
import regex as re
from collections import defaultdict
from cs336_basics.utils import log_print
import tqdm
import multiprocessing as mp
from multiprocessing import Pool

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
INITIAL_VOCAB_SIZE = 2**8
SPLIT_SPECIAL_TOKEN = "<|endoftext|>".encode("utf-8")

tqdm.tqdm.set_lock(mp.RLock())

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

def _pre_tokenize_worker(
    worker_indx: int,
    input_path: str | os.PathLike,
    special_tokens: list[str],
    start: int,
    end: int,
) -> dict[tuple[bytes, ...], int]:
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    
    # split by special tokens
    escaped_special_tokens = [re.escape(special_token) for special_token in special_tokens]
    documents = re.split("|".join(escaped_special_tokens), chunk)

    pre_tokenized_document: dict[tuple[bytes, ...], int] = defaultdict(int)

    for document in tqdm.tqdm(
        documents,
        position=worker_indx,
        desc=f"worker {worker_indx}",
        leave=False,
    ):
        for match in re.finditer(PAT, document):
            token = match.group()
            token_encoded = token.encode("utf-8")
            token_encoded = tuple([bytes([b]) for b in token_encoded])
            pre_tokenized_document[token_encoded] += 1
    
    return pre_tokenized_document


def _pre_tokenize(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int,
    num_splits: int,
) -> dict[tuple[bytes, ...], int]:

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_splits, SPLIT_SPECIAL_TOKEN)

    ranges = [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]

    pre_tokenize_worker_args = [
        (idx, str(input_path), special_tokens, start, end)
        for idx, (start, end) in enumerate(ranges)
    ]

    with Pool(processes=num_processes) as pool:
        pre_tokenized_subdocuments = pool.starmap(_pre_tokenize_worker, pre_tokenize_worker_args)
    
    # aggregate results
    pre_tokenized_document: dict[tuple[bytes, ...], int] = defaultdict(int)
    for pre_tokenized_subdocument in tqdm.tqdm(pre_tokenized_subdocuments):
        for key, value in pre_tokenized_subdocument.items():
            pre_tokenized_document[key] += value

    return pre_tokenized_document


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    log_print(f"Entered train_bpe.")

    num_processes = kwargs.get("num_processes", 4)
    num_splits = kwargs.get("num_splits", 8)

    # initialize vocabulary
    vocab: dict[int, bytes] = {}
    for i, special_token in enumerate(special_tokens):
        vocab[i] = special_token.encode("utf-8")
    offset = len(special_tokens)
    for i in range(INITIAL_VOCAB_SIZE):
        vocab[offset + i] = bytes([i])

    # initialize merges
    merges = []

    # now pre-tokenize each document
    log_print(f"Pre-tokenizing documents...")
    pre_tokenized_document = _pre_tokenize(
        input_path,
        special_tokens,
        num_processes=num_processes,
        num_splits=num_splits,
    )
    log_print(f"Done pre-tokenizing documents.")
    
    log_print(f"Building initial byte pair dict...")

    byte_pair_counts: dict[tuple[bytes, ...], int] = defaultdict(int)
    byte_pair_to_token_cache: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)

    for key, value in pre_tokenized_document.items():
        for i in range(len(key)-1):
            byte_pair = (key[i], key[i+1])
            byte_pair_counts[byte_pair] += value
            byte_pair_to_token_cache[byte_pair].add(key)
    
    log_print(f"Done building initial byte pair dict.")
    
    num_merges_to_make = vocab_size - len(vocab)
    for _ in tqdm.tqdm(range(num_merges_to_make)):

        top_byte_pair = None
        for byte_pair, count in byte_pair_counts.items():
            if (top_byte_pair is None or
                count > byte_pair_counts[top_byte_pair] or
                (count == byte_pair_counts[top_byte_pair] and byte_pair > top_byte_pair)):
                top_byte_pair = byte_pair

        # log_print(f"Done counting byte pairs.")
        vocab[len(vocab)] = top_byte_pair[0] + top_byte_pair[1]
        merges.append(top_byte_pair)

        old_toks = byte_pair_to_token_cache[top_byte_pair].copy()
        for old_tok in old_toks:
            count = pre_tokenized_document[old_tok]

            new_tok = []
            i = 0
            while i < len(old_tok):
                if i+1 < len(old_tok) and (old_tok[i], old_tok[i+1]) == top_byte_pair:
                    new_tok.append(old_tok[i] + old_tok[i+1])
                    i += 2
                else:
                    new_tok.append(old_tok[i])
                    i += 1
            
            pre_tokenized_document.pop(old_tok)
            
            for j in range(len(old_tok) - 1):
                old_pair = (old_tok[j], old_tok[j+1])
                byte_pair_counts[old_pair] -= count
                byte_pair_to_token_cache[old_pair].discard(old_tok)

            pre_tokenized_document[tuple(new_tok)] += count
            for j in range(len(new_tok) - 1):
                new_pair = (new_tok[j], new_tok[j+1])
                byte_pair_counts[new_pair] += count
                byte_pair_to_token_cache[new_pair].add(tuple(new_tok))

    log_print(f"Exiting train_bpe.")
    return vocab, merges