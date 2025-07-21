import torch
import jaxtyping
from torch import Tensor
from jaxtyping import Float, Bool, Int
import einops

class Linear(torch.nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        std = 2/(self.weight.shape[0] + self.weight.shape[1])
        torch.nn.init.trunc_normal_(
            tensor=self.weight,
            mean=0, std=std,
            a=-3*std, b=3*std,
        )
    
    def forward(self, x: Float[Tensor, '... d_in']) -> Float[Tensor, '... d_out']:
        return einops.einsum(self.weight, x, 'd_out d_in, ... d_in -> ... d_out')

class Embedding(torch.nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.weight = torch.nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        torch.nn.init.trunc_normal_(
            tensor=self.weight,
            mean=0, std=1,
            a=-3, b=3
        )

    def forward(self, x: Int[Tensor, 'batch seq']) -> Float[Tensor, 'batch seq d_model']:
        return self.weight[x]

        # batch, seq = x.shape
        # n_vocab, d_model = self.W_E.shape

        # W_E = self.W_E.unsqueeze(0).unsqueeze(0).expand(batch, seq, -1, -1)
        # x = x.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, d_model)
        # out = torch.gather(W_E, dim=2, index=x).squeeze(2)
        # return out

class RMSNorm(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.weight = torch.nn.Parameter(
            torch.empty((d_model), device=device, dtype=dtype)
        )
        torch.nn.init.ones_(tensor=self.weight)

        self.eps = torch.tensor(eps, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, '... d_model']) -> Float[Tensor, '... d_model']:
        # upcast to prevent overflow when we square the input
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(x.pow(2).mean(dim=-1) + self.eps).unsqueeze(-1)
        result = (x / rms) * self.weight

        return result.to(in_dtype)


def silu(x: Float[Tensor, '... d']):
    return x / (1 + torch.exp(-x))

class SwiGLU(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, '... d_model']) -> Float[Tensor, '... d_model']:
        gate = silu(self.w1(x))
        val  = self.w3(x)
        gated_val = gate * val
        out_proj = self.w2(gated_val)
        return out_proj

class RotaryPositionalEmbedding(torch.nn.Module):

    '''
    x_0 = x_0 cos - x_1 sin -> x_0 = x_0 cos + (-x_1) sin
    x_1 = x_0 sin + x_1 cos -> x_1 = x_1 cos + x_0 cos
    '''

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

        self._build_rope_cache()

    def _build_rope_cache(self):
        freqs = torch.arange(0, self.d_k, 2, device=self.device, dtype=self.dtype) # d_k/2
        inv_freq = torch.pow(self.theta, -freqs/self.d_k) # d_k/2

        positions = torch.arange(0, self.max_seq_len, device=self.device, dtype=self.dtype) # seq

        thetas = positions.unsqueeze(-1) * inv_freq.unsqueeze(0) # seq d_k/2
        sin = einops.repeat(torch.sin(thetas), 'seq d_k_2 -> seq (d_k_2 2)') # seq d_k
        cos = einops.repeat(torch.cos(thetas), 'seq d_k_2 -> seq (d_k_2 2)') # seq d_k

        self.register_buffer('sin', sin.to(dtype=self.dtype, device=self.device), persistent=False)
        self.register_buffer('cos', cos.to(dtype=self.dtype, device=self.device), persistent=False)
    
    @staticmethod
    def _get_flipped_pairs(x: Float[Tensor, '... d_k']):
        odds  = x[..., 1::2]
        evens = x[..., 0::2]
        return einops.rearrange(
            torch.stack((-odds, evens), dim=-2),
            '... two d_k_2 -> ... (d_k_2 two)'
        )

    def forward(
        self,
        x: Float[Tensor, '... seq d_k'],
        token_positions: Int[Tensor, '... seq']
    ) -> Float[Tensor, '... seq d_k']:
        return x * self.cos[token_positions] + self._get_flipped_pairs(x) * self.sin[token_positions]

def softmax(x: Float[Tensor, '... d'], dim: int) -> Float[Tensor, '... d']:
    x = x - einops.reduce(x, '... d -> ... 1', reduction='max')
    return torch.exp(x) / torch.exp(x).sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(
    q: Float[Tensor, '... seq_len d_q'],
    k: Float[Tensor, '... seq_len d_k'],
    v: Float[Tensor, '... seq_len d_v'],
    mask: Bool[Tensor, '... seq_len seq_len'],
) -> Float[Tensor, '... seq_len d_v']:

    scores = einops.einsum(q, k, '... seq_q d_head, ... seq_k d_head -> ... seq_q seq_k')
    scores = scores / (k.shape[-1]**0.5)
    scores = torch.masked_fill(scores, ~mask, float('-inf'))

    weights = softmax(scores, dim=-1)

    weighted_v = einops.einsum(weights, v, '... seq_q seq_k, ... seq_k d_v -> ... seq_q d_v')

    return weighted_v


class MultiHeadSelfAttention(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.rope = rope

        self.q_proj = Linear(in_features=d_model, out_features=self.num_heads*self.d_head, device=device, dtype=dtype)
        self.k_proj = Linear(in_features=d_model, out_features=self.num_heads*self.d_head, device=device, dtype=dtype)
        self.v_proj = Linear(in_features=d_model, out_features=self.num_heads*self.d_head, device=device, dtype=dtype)
        self.output_proj = Linear(in_features=self.num_heads*self.d_head, out_features=d_model, device=device, dtype=dtype)
        
    def forward(
        self,
        x: Float[Tensor, '... seq d_model'],
        token_positions: Int[Tensor, '... seq'] | None = None
    ) -> Float[Tensor, '... seq d_model']:

        q = self.q_proj(x)
        q = einops.rearrange(q, '... seq (num_heads d_head) -> ... num_heads seq d_head', num_heads=self.num_heads, d_head=self.d_head)
        k = self.k_proj(x)
        k = einops.rearrange(k, '... seq (num_heads d_head) -> ... num_heads seq d_head', num_heads=self.num_heads, d_head=self.d_head)
        v = self.v_proj(x)
        v = einops.rearrange(v, '... seq (num_heads d_head) -> ... num_heads seq d_head', num_heads=self.num_heads, d_head=self.d_head)

        if self.rope:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        seq = x.shape[-2]
        mask = torch.tril(torch.ones((seq, seq))).bool().to(device=q.device)
        weighted_v = scaled_dot_product_attention(q, k, v, mask) # ... num_heads seq d_head
        weighted_v = einops.rearrange(weighted_v, '... batch num_heads seq d_head -> ... batch seq (num_heads d_head)', num_heads=self.num_heads, d_head=self.d_head)

        out = self.output_proj(weighted_v)
        return out

class TransformerBlock(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RotaryPositionalEmbedding,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):

        super().__init__()

        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
    
    def forward(
        self,
        x: Float[Tensor, '... seq d_model'],
        token_positions: Int[Tensor, '... seq']
    ) -> Float[Tensor, '... seq d_model']:
        
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))

        return x

class TransformerLM(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        rope_theta: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        
        super().__init__()

        self.device = device
        self.dtype = dtype

        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)

        d_k = d_model // num_heads
        self.rope = RotaryPositionalEmbedding(theta=rope_theta, d_k=d_k, max_seq_len=context_length, device=device, dtype=dtype)

        self.layers = torch.nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, self.rope, device=device, dtype=dtype)
                for _ in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: Int[Tensor, '... seq']) -> Float[Tensor, '... seq n_vocab']:

        token_positions = torch.arange(token_ids.shape[-1])

        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, token_positions)
        
        logits = self.lm_head(self.ln_final(x))

        return logits
