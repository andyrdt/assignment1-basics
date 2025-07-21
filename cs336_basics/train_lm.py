import torch
import einops
import math
from torch import Tensor
from jaxtyping import Float, Int
from typing import Optional, Callable, Tuple, IO, BinaryIO
import numpy as np
import random
import typing
from os import PathLike
from functools import partial

from .transformer import softmax

def log_softmax(
    logits: Float[Tensor, '... d_vocab'],
    dim: int,
) -> Float[Tensor, '... d_vocab']:
    
    logits = logits - logits.max(dim=dim, keepdim=True).values
    normalizers = torch.log(torch.exp(logits).sum(dim=-1, keepdim=True))
    out = logits - normalizers
    return out

def cross_entropy(
    logits: Float[Tensor, 'batch d_vocab'],
    targets: Int[Tensor, 'batch']
) -> Float[Tensor, '']:

    log_probs = log_softmax(logits, dim=-1)

    target_log_probs = torch.gather(
        input=log_probs,
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    losses = -target_log_probs

    return losses.mean()


def compute_perplexity(
    logits: Float[Tensor, 'batch d_vocab'],
    targets: Int[Tensor, 'batch']
) -> Float[Tensor, '']:

    return torch.exp(cross_entropy(logits, targets))

class AdamW(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        lr: float =1e-3,
        betas: Tuple[float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
        }

        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable]= None):

        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group['lr']
            betas = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get('t', 1)
                m = state.get('m', 0)
                v = state.get('v', 0)

                g = p.grad.data
                m = betas[0] * m + (1 - betas[0]) * g
                v = betas[1] * v + (1 - betas[1]) * torch.pow(g, 2)

                lr_adjusted = lr * math.sqrt(1 - math.pow(betas[1], t)) / (1 - math.pow(betas[0], t))
                
                p.data -= lr_adjusted * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data

                state['t'] = t + 1
                state['m'] = m
                state['v'] = v
        
        return loss

class CosineAnnealingLRScheduler:

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_max: float,
        lr_min: float,
        t_warmup: int,
        t_cosine: int,
    ):
        self.optimizer = optimizer
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.t_warmup = t_warmup
        self.t_cosine = t_cosine
        self.step_count = 0

        self.get_lr = partial(
            cosine_annealing_lr_schedule,
            lr_max=self.lr_max,
            lr_min=self.lr_min,
            t_warmup=self.t_warmup,
            t_cosine=self.t_cosine,
        )
    
    def step(self):
        self.step_count = self.step_count + 1
        lr = self.get_lr(self.step_count)
        for group in self.optimizer.param_groups:
            group['lr'] = lr

def cosine_annealing_lr_schedule(
    t: int,
    lr_max: float,
    lr_min: float,
    t_warmup: int,
    t_cosine: int,
) -> float:

    # warmup
    if t < t_warmup:
        return (t / t_warmup) * lr_max

    # cosine annealing
    if t < t_cosine:
        return lr_min + 0.5 * (1 + math.cos(math.pi * (t - t_warmup) / (t_cosine - t_warmup))) * (lr_max - lr_min)

    # post-annealing
    return lr_min

class LinearLRScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_max: float,
        lr_min: float,
        t_warmup: int,
        t_max: int,
    ):
        self.optimizer = optimizer
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.t_warmup = t_warmup
        self.t_max = t_max
        self.step_count = 0

        self.get_lr = partial(
            linear_lr_schedule,
            lr_max=self.lr_max,
            lr_min=self.lr_min,
            t_warmup=self.t_warmup,
            t_max=self.t_max,
        )

    def step(self):
        self.step_count = self.step_count + 1
        lr = self.get_lr(self.step_count)

def linear_lr_schedule(
    t: int,
    lr_max: float,
    lr_min: float,
    t_warmup: int,
    t_max: int,
) -> float:
    if t < t_warmup:
        return lr_min + (lr_max - lr_min) * (t / t_warmup)
    return lr_min + (lr_max - lr_min) * (1 - (t - t_warmup) / (t_max - t_warmup))

def gradient_clipping(parameters, max_l2_norm, eps=1e-6):
    
    sum_of_squares = 0.0
    for p in parameters:
        if p.grad is None:
            continue

        sum_of_squares = sum_of_squares + (p.grad**2).sum()
        
    l2_norm = sum_of_squares**0.5

    if l2_norm > max_l2_norm:
        scale_factor = (max_l2_norm / (l2_norm + eps))
        for p in parameters:
            if p.grad is None:
                continue
            
            p.grad = p.grad * scale_factor

def get_batch(
    x,
    batch_size,
    context_length,
    device
):
    start_indices = torch.randint(low=0, high=x.shape[0] - context_length, size=(batch_size,))
    token_indices = einops.repeat(start_indices, 'batch_size -> batch_size context_length', context_length=context_length).contiguous()
    token_indices = token_indices + torch.arange(0, context_length).unsqueeze(dim=0)

    label_indices = token_indices.clone() + 1

    tokens = torch.tensor(x[token_indices.numpy()], dtype=torch.long, device=device)
    labels = torch.tensor(x[label_indices.numpy()], dtype=torch.long, device=device)

    return (tokens, labels)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | PathLike | BinaryIO | IO[bytes]):
    obj = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration
    }

    torch.save(obj, out)

def load_checkpoint(
    src: str | PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    obj = torch.load(src, weights_only=False)

    model.load_state_dict(obj['model'])
    optimizer.load_state_dict(obj['optimizer'])

    return obj['iteration']


