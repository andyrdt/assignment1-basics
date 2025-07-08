import codecs
import os
import heapq
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
WORKER_CHUNK_SIZE = 1 << 20  # 1MB

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
    chunk_size: int = WORKER_CHUNK_SIZE,
) -> dict[tuple[bytes, ...], int]:

    splitter = re.compile("|".join(map(re.escape, special_tokens)))
    incremental_decoder = codecs.getincrementaldecoder("utf-8")()
    counts: dict[tuple[bytes, ...], int] = defaultdict(int)

    with open(input_path, "rb") as f:
        to_read = end - start
        f.seek(start)

        carry = ""

        with tqdm.tqdm(
            total=to_read,
            position=worker_indx,
            desc=f"worker {worker_indx}",
            leave=False,
        ) as pbar:
            while to_read > 0:
                chunk = f.read(min(chunk_size, to_read))
                to_read -= len(chunk)
                pbar.update(len(chunk))

                decoded_chunk = incremental_decoder.decode(chunk)
                decoded_chunk = carry + decoded_chunk

                *docs, carry = splitter.split(decoded_chunk)

                for doc in docs:
                    for match in re.finditer(PAT, doc):
                        token_encoded = match.group().encode("utf-8")
                        token_key = tuple([bytes([b]) for b in token_encoded])
                        counts[token_key] += 1 

        leftover = incremental_decoder.decode(bytes(), final=True)
        if leftover:
            tail_doc = carry + leftover
        else:
            tail_doc = carry
        
        if tail_doc:
            for match in re.finditer(PAT, tail_doc):
                token_encoded = match.group().encode("utf-8")
                token_key = tuple(bytes([b]) for b in token_encoded)
                counts[token_key] += 1

    return counts

def _pre_tokenize(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int,
) -> dict[tuple[bytes, ...], int]:
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, SPLIT_SPECIAL_TOKEN)

    ranges = [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]

    pre_tokenize_worker_args = [
        (idx, str(input_path), special_tokens, start, end)
        for idx, (start, end) in enumerate(ranges)
    ]

    with Pool(processes=num_processes) as pool:
        pre_tokenized_subdocuments = pool.starmap(_pre_tokenize_worker, pre_tokenize_worker_args)
    
    # aggregate results
    counts: dict[tuple[bytes, ...], int] = defaultdict(int)
    for pre_tokenized_subdocument in tqdm.tqdm(pre_tokenized_subdocuments):
        for key, value in pre_tokenized_subdocument.items():
            counts[key] += value

    return counts

class BPEStats:

    class _HeapItem:
        def __init__(self, count: int, pair: tuple[bytes, bytes]):
            self.count = count
            self.pair = pair

        def __lt__(self, other) -> bool:
            if self.count != other.count:
                return self.count > other.count # larger count wins
            return self.pair > other.pair # larger pair wins on tie

    def __init__(self, token_counts: dict[tuple[bytes, ...], int], max_num_merges: int):
        self.token_counts: dict[tuple[bytes, ...], int] = token_counts
        self.pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
        self.pair_to_tokens: dict[tuple[bytes, bytes], tuple[bytes, ...]] = defaultdict(set)
        self._build_initial_stats()
        
        self.max_num_merges = max_num_merges
        self.heap: list[tuple[int, tuple[bytes, bytes], tuple[bytes, bytes]]] = []
        self._build_initial_heap()
    
    def _build_initial_stats(self):
        for token, count in self.token_counts.items():
            for i in range(len(token) - 1):
                pair = (token[i], token[i+1])
                self.pair_counts[pair] += count
                self.pair_to_tokens[pair].add(token)

    def _build_initial_heap(self):
        for pair, count in self.pair_counts.items():
            heapq.heappush(self.heap, self._HeapItem(count, pair))
    
    def get_top_pair(self) -> tuple[bytes, bytes] | None:
        while self.heap:
            item = self.heap[0]
            count, pair = item.count, item.pair
            if count == self.pair_counts[pair] and self.pair_counts[pair] > 0:
                # heap entry is not stale
                return pair
            else:
                # heap entry is stale (we add heap entries lazily - see merge_pair)
                # remove the entry, and continue until we hit a valid one
                heapq.heappop(self.heap)
        return None
    
    def merge_pair(self, pair: tuple[bytes, bytes]) -> None:

        tokens_to_modify = self.pair_to_tokens[pair].copy()
        touched = set()
        previous_pair_counts = self.pair_counts.copy()

        for idx, old_token in enumerate(tokens_to_modify):
            count = self.token_counts.pop(old_token)

            new_token = []
            i = 0
            while i < len(old_token):
                if i+1 < len(old_token) and (old_token[i], old_token[i+1]) == pair:
                    new_token.append(old_token[i] + old_token[i+1])
                    i += 2
                else:
                    new_token.append(old_token[i])
                    i += 1
            new_token = tuple(new_token)
            self.token_counts[new_token] += count

            # remove old token contributions
            for j in range(len(old_token) - 1):
                old_pair = (old_token[j], old_token[j+1])
                self.pair_counts[old_pair] -= count
                self.pair_to_tokens[old_pair].discard(old_token)
                touched.add(old_pair)

            # add new token contributions
            for j in range(len(new_token) - 1):
                new_pair = (new_token[j], new_token[j+1])
                self.pair_counts[new_pair] += count
                self.pair_to_tokens[new_pair].add(new_token)
                touched.add(new_pair)

        for pair in touched:
            updated_count = self.pair_counts[pair]
            previous_count = previous_pair_counts[pair]
            if updated_count != previous_count:
                heapq.heappush(self.heap, self._HeapItem(updated_count, pair))
            
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    num_processes = kwargs.get("num_processes", 8)

    # initialize vocabulary
    vocab: dict[int, bytes] = {}
    for i, special_token in enumerate(special_tokens):
        vocab[i] = special_token.encode("utf-8")
    offset = len(special_tokens)
    for i in range(INITIAL_VOCAB_SIZE):
        vocab[offset + i] = bytes([i])

    merges = []
    merges_needed = vocab_size - len(vocab)

    log_print("Pre-tokenizing...")    
    token_counts = _pre_tokenize(input_path, special_tokens, num_processes)
    log_print("Done pre-tokenizing.")

    log_print("Constructing BPEStats")
    stats = BPEStats(token_counts, merges_needed)
    log_print("Done constructing BPEStats")

    log_print("Merging pairs...")
    for _ in tqdm.tqdm(range(merges_needed)):
        top_pair = stats.get_top_pair()
        vocab[len(vocab)] = top_pair[0] + top_pair[1]
        merges.append(top_pair)
        stats.merge_pair(top_pair)
    log_print("Done merging pairs.")

    return vocab, merges
