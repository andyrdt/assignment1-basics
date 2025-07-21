
import datetime
import time
import torch
from dataclasses import dataclass, asdict
from cs336_basics.transformer import TransformerLM
from cs336_basics.train_lm import AdamW, CosineAnnealingLRScheduler, gradient_clipping, get_batch, save_checkpoint, load_checkpoint, cross_entropy, compute_perplexity
from functools import partial
import numpy as np
import tqdm
import wandb
import os
import json
import einops

@dataclass
class TrainLMConfig:
    # OUTPUT
    output_dir: str = "output/tinystories"

    # TOKENIZED INPUT
    vocab_size: int = 10_000
    train_tok_path: str = "results/tinystories/train.npy"
    valid_tok_path: str = "results/tinystories/valid.npy"

    # OPTIMIZER
    lr_max: float = 1e-3
    lr_min: float = 1e-6
    lr_warmup: float = 0.05

    adamw_weight_decay: float = 0.01
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.999

    max_grad_norm: float | None = None

    # MODEL
    d_model: int = 512
    d_ff: int = 1344
    rope_theta: float = 10000
    num_layers: int = 4
    num_heads: int = 16
    dtype: str = "bfloat16"

    # TRAINING
    batch_size: int = 32
    context_length: int = 256
    # total_tokens: int = 327_680_000
    total_tokens: int = 50_000_000

    # WANDB
    log_to_wandb: bool = True
    wandb_project: str = "cs336-assignment1"
    wandb_name_suffix: str = ""

    # SAVE / LOG
    log_every_n_steps: int = 10
    validate_every_n_steps: int = 200
    save_every_n_steps: int = 1000
    n_validation_tokens: int = 500_000
    n_validation_tokens_final: int = 5_000_000

def main():
    config = TrainLMConfig()
    print(config)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(config.output_dir, timestamp)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(asdict(config), f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if config.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif config.dtype == "float32":
        dtype = torch.float32
    else:
        raise ValueError(f"Invalid dtype: {config.dtype}")

    if device == "cpu":
        print("WARNING: Running on CPU. This will be super slow.")

    training_toks = np.load(config.train_tok_path, mmap_mode="r")
    validation_toks = np.load(config.valid_tok_path, mmap_mode="r")

    model = TransformerLM(
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        num_layers=config.num_layers,
        rope_theta=config.rope_theta,
        device=device,
        dtype=dtype,
    )

    n_steps = config.total_tokens // (config.batch_size * config.context_length)
    print(f"Total steps: {n_steps}")
    print(f"Total tokens: {n_steps * config.batch_size * config.context_length} ({n_steps} * {config.batch_size} * {config.context_length})")

    optimizer = AdamW(
        params=model.parameters(),
        lr=config.lr_min,
        betas=(config.adamw_beta1, config.adamw_beta2),
        weight_decay=config.adamw_weight_decay,
    )

    lr_warmup_steps = int(config.lr_warmup * n_steps)

    lr_scheduler = CosineAnnealingLRScheduler(
        optimizer=optimizer,
        lr_max=config.lr_max,
        lr_min=config.lr_min,
        t_warmup=lr_warmup_steps,
        t_cosine=n_steps - lr_warmup_steps,
    )

    if config.log_to_wandb:
        _create_wandb_run(config, timestamp)
    
    start_time = time.time()

    for step in tqdm.tqdm(range(n_steps)):
        optimizer.zero_grad()

        tokens, targets = get_batch(training_toks, config.batch_size, config.context_length, device=device)
        
        logits = model(tokens)
        loss = cross_entropy(logits, targets)
        loss.backward()

        if config.max_grad_norm is not None:
            gradient_clipping(model.parameters(), config.max_grad_norm)

        optimizer.step()
        lr_scheduler.step()

        if step % config.log_every_n_steps == 0:
            grad_norm = _compute_gradient_norm(model)
            weight_norm = _compute_weight_norm(model)
            perplexity = compute_perplexity(logits, targets)
            learning_rate = optimizer.param_groups[0]["lr"]
            tokens_processed = step * config.batch_size * config.context_length
            current_time = time.time()
            time_elapsed = current_time - start_time

            if config.log_to_wandb:
                wandb.log({
                    "loss": loss.item(),
                    "learning_rate": learning_rate,
                    "grad_norm": grad_norm.item(),
                    "weight_norm": weight_norm.item(),
                    "perplexity": perplexity.item(),
                    "time_elapsed": time_elapsed,
                    "tokens_processed": tokens_processed,
                }, step=step)
            else:
                print(f"Step {step}")
                print(f"Loss: {loss.item()}")
                print(f"Learning rate: {learning_rate}")
                print(f"Grad norm: {grad_norm}")
                print(f"Weight norm: {weight_norm}")
                print(f"Perplexity: {perplexity}")
                print(f"Time elapsed: {time_elapsed}")
                print(f"==============================================\n")
        
        if step % config.validate_every_n_steps == 0:
            validation_loss = _validate(
                model=model,
                validation_toks=validation_toks,
                batch_size=config.batch_size,
                context_length=config.context_length,
                n_tokens=config.n_validation_tokens,
                device=device,
            )
            if config.log_to_wandb:
                wandb.log({
                    "validation_loss": validation_loss.item(),
                }, step=step)

        if step % config.save_every_n_steps == 0:
            save_path = os.path.join(out_dir, f"checkpoint_{step}.pt")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=step,
                out=save_path,
            )
    
    # evaluate final model
    validation_loss = _validate(
        model=model,
        validation_toks=validation_toks,
        batch_size=config.batch_size,
        context_length=config.context_length,
        n_tokens=config.n_validation_tokens_final,
        device=device,
    )

    if config.log_to_wandb:
        wandb.log({
            "validation_loss_final": validation_loss.item(),
        }, step=step)

    save_path = os.path.join(out_dir, f"final.pt")
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=step,
        out=save_path,
    )

def _validate(
    model: torch.nn.Module,
    validation_toks: np.ndarray,
    batch_size: int,
    context_length: int,
    n_tokens: int,
    device: str,
):
    model.eval()

    n_batches = n_tokens // (batch_size * context_length)
    total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    total_tokens = torch.tensor(0, device=device, dtype=torch.int32)

    for i in tqdm.tqdm(range(n_batches)):
        tokens, labels = get_batch(
            validation_toks,
            batch_size,
            context_length,
            device
        )

        with torch.no_grad():
            logits = model(tokens)

        loss = cross_entropy(logits, labels)
        total_loss += loss.to(dtype=torch.float32) * batch_size * context_length
        total_tokens += batch_size * context_length

    model.train()

    return total_loss / total_tokens


def _compute_gradient_norm(model: torch.nn.Module):
    sum_of_squares = torch.tensor(0.0, device=model.device, dtype=torch.float32)
    for p in model.parameters():
        if p.grad is None:
            continue

        sum_of_squares += (p.grad**2).to(dtype=torch.float32).sum()

    l2_norm = sum_of_squares**0.5

    return l2_norm

def _compute_weight_norm(model: torch.nn.Module):
    sum_of_squares = torch.tensor(0.0, device=model.device, dtype=torch.float32)
    for p in model.parameters():
        if p.data is None:
            continue

        sum_of_squares += (p.data**2).to(dtype=torch.float32).sum()

    l2_norm = sum_of_squares**0.5

    return l2_norm

def _create_wandb_run(config: TrainLMConfig, timestamp: str):
    params_for_name = [
        f"lr={config.lr_max}",
        f"bs={config.batch_size}",
    ]
    params_in_name = "_".join(params_for_name)
    wandb_name = f"{timestamp}_{params_in_name}_{config.wandb_name_suffix}"

    wandb.init(
        project=config.wandb_project,
        name=wandb_name,
        config=config,
    )

if __name__ == "__main__":
    main()
