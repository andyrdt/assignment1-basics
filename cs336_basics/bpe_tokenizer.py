
from typing import Iterable, Iterator
import regex as re
from tests.common import gpt2_bytes_to_unicode
import json

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BPETokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.vocab_inv = {v: k for k, v in self.vocab.items()}
        self.merges = merges
        # Create a dictionary for O(1) merge lookups
        self.merge_ranks = {pair: idx for idx, pair in enumerate(merges)}
        self.special_tokens = special_tokens or []
        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            self.special_tokens_splitter = re.compile("|".join(map(re.escape, sorted_special_tokens)))
        else:
            self.special_tokens_splitter = re.compile(r'(?!)') # matches nothing
        self.pat_splitter = re.compile(PAT)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ):
        bytes_to_unicode_dict = gpt2_bytes_to_unicode()
        unicode_to_bytes_dict = {v: k for k, v in bytes_to_unicode_dict.items()}

        with open(vocab_filepath) as f_vocab:
            vocab_raw = json.load(f_vocab)
        with open(merges_filepath) as f_merges:
            merges_raw = json.load(f_merges)

        vocab = {
            v: bytes(unicode_to_bytes_dict[c] for c in k)
            for k, v in vocab_raw.items()
        }

        merges = [
            (
                bytes(unicode_to_bytes_dict[c] for c in merge[0]),
                bytes(unicode_to_bytes_dict[c] for c in merge[1])
            )
            for merge in merges_raw
        ]

        return cls(vocab, merges, special_tokens)

    def _pretokenize(self, text: str) -> list[tuple[bytes]]:
        docs = self.special_tokens_splitter.split(text)
        special_tokens = self.special_tokens_splitter.findall(text)

        special_tokens = special_tokens + [None]
        assert len(docs) == len(special_tokens)

        tokens = []

        for doc, special_token in zip(docs, special_tokens):
            for match in self.pat_splitter.finditer(doc):
                token_encoded = match.group().encode("utf-8")
                token = tuple([bytes([b]) for b in token_encoded])
                tokens.append(token)
            if special_token:
                token_encoded = special_token.encode("utf-8")
                token = (token_encoded,)
                tokens.append(token)

        return tokens
    
    def _bpe_merge_token(self, token: tuple[bytes]):
        if len(token) <= 1:
            return token
            
        token = list(token)
        
        # Apply merges in order until no more merges can be applied
        while True:
            pairs = [(token[i], token[i+1]) for i in range(len(token) - 1)]
            
            # Find the earliest merge that can be applied using O(1) lookup
            best_merge_rank = None
            best_pair_idx = None
            
            for pair_idx, pair in enumerate(pairs):
                if pair in self.merge_ranks:
                    merge_rank = self.merge_ranks[pair]
                    if best_merge_rank is None or merge_rank < best_merge_rank:
                        best_merge_rank = merge_rank
                        best_pair_idx = pair_idx
            
            # If no merge found, we're done
            if best_merge_rank is None:
                break
                
            # Apply the best merge
            token[best_pair_idx] = token[best_pair_idx] + token[best_pair_idx + 1]
            token.pop(best_pair_idx + 1)
        
        return tuple(token)

    def encode(self, text: str) -> list[int]:
        tokens = self._pretokenize(text)

        bpe_merged_tokens = []
        for token in tokens:
            bpe_merged_tokens.append(self._bpe_merge_token(token))

        encoded_tokens = []
        for bpe_merged_token in bpe_merged_tokens:
            for sub_token in bpe_merged_token:
                encoded_tokens.append(self.vocab_inv[sub_token])

        return encoded_tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        for chunk in iterable:
            buffer += chunk

            *docs, remainder = self.special_tokens_splitter.split(buffer)
            special_tokens = self.special_tokens_splitter.findall(buffer)

            assert len(docs) == len(special_tokens)

            for doc, special_token in zip(docs, special_tokens):
                toks = self.encode(doc)
                toks += self.encode(special_token)

                yield from toks

            buffer = remainder
        
        if buffer:
            yield from self.encode(buffer)

    def decode(self, ids: list[int]) -> str:
        result_bytes = []
        for id in ids:
            result_bytes.append(self.vocab[id])
        return b"".join(result_bytes).decode('utf-8', errors='replace')
