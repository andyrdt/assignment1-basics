from cs336_basics.train_bpe import train_bpe
import json
import os

input_path = 'data/TinyStoriesV2-GPT4-train.txt'
vocab_size = 10_000
special_tokens = ['<|endoftext|>']
output_dir = 'data/tinystories'

vocab, merges = train_bpe(
    input_path,
    vocab_size,
    special_tokens,
    num_processes=8,
    num_splits=8,
)

# print(vocab)
vocab_serialized = {k: repr(v) for k, v in vocab.items()}
merges_serialized = [tuple(repr(m) for m in merge) for merge in merges]

os.makedirs(output_dir, exist_ok=True)

with open(f"{output_dir}/vocab.json", 'w') as f:
    f.write(json.dumps(vocab_serialized, indent=2))
with open(f"{output_dir}/merges.json", 'w') as f:
    f.write(json.dumps(merges_serialized, indent=2))

# sort by length of vocab
vocab_dict = sorted(vocab_serialized.items(), key=lambda x: len(x[1]))
print("Longest tokens:")
for i in range(20):
    print(vocab_dict[-i-1])
