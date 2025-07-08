import argparse
import json
import os
import cProfile
import pstats
from cs336_basics.train_bpe import train_bpe
from tests.common import gpt2_bytes_to_unicode


def main():

    DATASET_CONFIGS = {
        'tinystories': {
            'input_path': 'data/TinyStoriesV2-GPT4-train.txt',
            'vocab_size': 10_000,
            'special_tokens': ['<|endoftext|>'],
            'output_dir': 'results/tinystories',
            'num_processes': 8,
        },
        'owt': {
            'input_path': 'data/owt_train.txt',
            'vocab_size': 32_000,
            'special_tokens': ['<|endoftext|>'],
            'output_dir': 'results/owt',
            'num_processes': 8,
        }
    }
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train BPE tokenizer with profiling',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--dataset', type=str, choices=['tinystories', 'owt'], 
                       required=True, help='Dataset to train on')
    parser.add_argument('--output-dir', type=str, help='Override output directory')
    parser.add_argument('--vocab-size', type=int, help='Override vocabulary size')
    parser.add_argument('--num-processes', type=int, help='Override number of processes')
    parser.add_argument('--no-profile', action='store_true', 
                       help='Disable profiling (faster but no performance data)')
    
    args = parser.parse_args()
    
    # Get dataset configuration
    config = DATASET_CONFIGS[args.dataset].copy()
    
    # Override with command line arguments if provided
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.vocab_size:
        config['vocab_size'] = args.vocab_size
    if args.num_processes:
        config['num_processes'] = args.num_processes
    
    print(f"Training BPE for {args.dataset} dataset")
    print(f"Configuration: {config}")
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Optional profiling setup
    pr = None
    if not args.no_profile:
        print("Profiling enabled (cProfile)...")
        pr = cProfile.Profile()
        pr.enable()
    
    # Train BPE
    print("Starting BPE training...")
    vocab, merges = train_bpe(
        input_path=config['input_path'],
        vocab_size=config['vocab_size'],
        special_tokens=config['special_tokens'],
        num_processes=config['num_processes']
    )
    
    # Stop profiling and save results
    if pr is not None:
        pr.disable()
        
        # Save profile data
        profile_path = os.path.join(config['output_dir'], f'train_bpe_{args.dataset}.prof')
        pr.dump_stats(profile_path)
        print(f"Profile data saved to: {profile_path}")
        
        # Print profiling stats
        print(f"\n=== Top 15 functions by cumulative time ===")
        stats = pstats.Stats(pr)
        stats.sort_stats('cumtime').print_stats(15)
    
    print("Training completed!")
    print(f"Final vocabulary size: {len(vocab):,}")
    print(f"Number of merges: {len(merges):,}")
    
    # Serialize and save vocab and merges
    print("Serializing vocabulary and merges...")

    bytes_to_unicode_dict = gpt2_bytes_to_unicode()
    
    def bytes_to_unicode(bytes_data):
        return ''.join([bytes_to_unicode_dict[byte] for byte in bytes_data])
    
    vocab_serialized = {bytes_to_unicode(v): k for k, v in vocab.items()}
    merges_serialized = [[bytes_to_unicode(m) for m in merge] for merge in merges]

    vocab_path = os.path.join(config['output_dir'], 'vocab.json')
    merges_path = os.path.join(config['output_dir'], 'merges.json')
    
    with open(vocab_path, 'w') as f:
        json.dump(vocab_serialized, f, indent=2)
    
    with open(merges_path, 'w') as f:
        json.dump(merges_serialized, f, indent=2)
    
    print(f"Vocab saved to: {vocab_path}")
    print(f"Merges saved to: {merges_path}")
    
    # Print longest tokens
    print_longest_tokens(vocab_serialized)
    
    # Print vocabulary statistics
    print_vocab_statistics(vocab, config['special_tokens'])


def print_longest_tokens(vocab_serialized: dict, top_n: int = 20):
    """Print the longest tokens in the vocabulary."""

    # Sort by token length (string length)
    vocab_by_length = sorted(vocab_serialized.items(), key=lambda x: len(x[0]), reverse=True)
    print(f"\nTop {top_n} longest tokens (by character length)")
    print("=" * 60)
    for i in range(min(top_n, len(vocab_by_length))):
        token, token_id = vocab_by_length[i]
        # Escape special characters for display
        display_token = repr(token)[1:-1]  # Remove outer quotes from repr
        print(f"{i+1:2d}. Token ID {token_id:5d}: '{display_token}' (length: {len(token)})")

def print_vocab_statistics(vocab: dict, special_tokens: list):
    """Print comprehensive vocabulary statistics."""
    
    total_tokens = len(vocab)
    special_token_count = len(special_tokens)
    byte_tokens = 256  # ASCII byte tokens
    learned_tokens = total_tokens - special_token_count - byte_tokens
    
    print("\nVocabulary Statistics")
    print("=" * 40)
    print(f"Total tokens:           {total_tokens:,}")
    print(f"Special tokens:         {special_token_count:,}")
    print(f"Byte-level tokens:      {byte_tokens:,}")
    print(f"Learned BPE tokens:     {learned_tokens:,}")
    print(f"Compression ratio:      {learned_tokens / total_tokens:.2%}")
    
    # Calculate token distribution
    if learned_tokens > 0:
        print("\nToken Distribution")
        print(f"Special tokens:         {special_token_count / total_tokens:6.2%}")
        print(f"Byte-level tokens:      {byte_tokens / total_tokens:6.2%}")
        print(f"Learned BPE tokens:     {learned_tokens / total_tokens:6.2%}")

if __name__ == "__main__":
    main() 