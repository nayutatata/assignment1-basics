import torch
import math
from typing import Iterable
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.T)

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.vocab_size = num_embeddings
        self.d_model = embedding_dim
        self.weight = torch.nn.Parameter(torch.empty((self.vocab_size, self.d_model), device=device, dtype=dtype))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RmsNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones((d_model,), device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        x_var = x.pow(2).mean(-1, keepdim=True) + self.eps
        res = x * x_var.rsqrt() * self.weight
        return res.to(in_dtype)

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.d_model = d_model
        self.d_ff = d_ff

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layer1_res = self.silu(self.w1(x))
        gate  = self.w3(x)
        after_gate = layer1_res * gate
        return self.w2(after_gate)

class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len
        thetas = torch.arange(0, self.d_k, 2, device=device) / self.d_k
        thetas = 1.0 / (self.theta ** thetas)
        assert thetas.shape == (self.d_k // 2,)
        seq_ids = torch.arange(0, self.max_seq_len, device=device)
        final_thetas = seq_ids[:, None] * thetas[None, :]
        self.register_buffer("cos_cache", torch.cos(final_thetas), persistent=False)
        self.register_buffer("sin_cache", torch.sin(final_thetas), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x1_cos = x1 * self.cos_cache[token_positions]
        x1_sin = x1 * self.sin_cache[token_positions]
        x2_cos = x2 * self.cos_cache[token_positions]
        x2_sin = x2 * self.sin_cache[token_positions]
        out = torch.stack([x1_cos - x2_sin, x1_sin + x2_cos], dim=-1).flatten(-2, -1)
        return out
    
def my_softmax(x: torch.Tensor, dim:int):
    max_v, _ = x.max(dim=dim, keepdim=True)
    x = (x - max_v).exp()
    x_esum = x.sum(dim=dim, keepdim=True)
    return x/x_esum

def my_scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor)->torch.Tensor:
    q_k = q @ k.transpose(dim0=-1, dim1=-2) / k.shape[-1]**0.5
    attn_mask = torch.where(mask, torch.tensor(0.0, device=q.device), torch.tensor(-float('inf'), device=q.device))
    q_k = q_k + attn_mask
    sm = my_softmax(q_k, -1)
    return sm @ v

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.head_dim = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.reshape(*q.shape[:-1], self.num_heads, self.head_dim).transpose(-3, -2)
        k = k.reshape(*k.shape[:-1], self.num_heads, self.head_dim).transpose(-3, -2)
        v = v.reshape(*v.shape[:-1], self.num_heads, self.head_dim).transpose(-3, -2)
        # here, shape = (batch_size, num_heads, seq_len, head_dim)
        mask = torch.tril(torch.ones((q.shape[-2], q.shape[-2]), dtype=torch.bool, device=q.device))
        atten = my_scaled_dot_product_attention(q,k,v, mask=mask).transpose(-3, -2).reshape(*x.shape)
        return atten @ self.output_proj.transpose(-1, -2)

class MhaWithRope(torch.nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len, theta, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.head_dim = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = RoPE(theta=theta, d_k=self.head_dim, max_seq_len=max_seq_len, device=device)

    def forward(self, x:torch.Tensor, token_positions:torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.reshape(*q.shape[:-1], self.num_heads, self.head_dim).transpose(-3, -2)
        k = k.reshape(*k.shape[:-1], self.num_heads, self.head_dim).transpose(-3, -2)
        v = v.reshape(*v.shape[:-1], self.num_heads, self.head_dim).transpose(-3, -2)
        q_rope = self.rope(q, token_positions)
        k_rope = self.rope(k, token_positions)
        mask = torch.tril(torch.ones((q.shape[-2], q.shape[-2]), dtype=torch.bool, device=q.device))
        atten = my_scaled_dot_product_attention(q_rope,k_rope,v, mask=mask).transpose(-3, -2).reshape(*x.shape)
        return atten @ self.output_proj.weight.transpose(-1, -2)

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, d_ff, num_heads, max_seq_len, theta, device=None, dtype=None):
        super().__init__()
        self.device = device
        self.attn = MhaWithRope(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.ln1 = RmsNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RmsNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.ln1(x)
        seq_len = x.shape[-2]
        token_positions = torch.arange(seq_len, device=self.device)
        attn_out = self.attn(x_norm, token_positions)
        x = x + attn_out
        x_norm = self.ln2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        return x

class TransformerLm(torch.nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, device=None, dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = torch.nn.ModuleList(
            [
                TransformerBlock(d_model, d_ff, num_heads, context_length, rope_theta, device=device, dtype=dtype)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RmsNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits

def my_cross_entropy(logits: torch.Tensor, target_token_ids: torch.Tensor) -> torch.Tensor:
    # logits: BSV, target: BS
    ls = logits.view(-1, logits.shape[-1]) # BV
    targets = target_token_ids.view(-1) # B
    target_prob = ls[torch.arange(ls.shape[0], device=targets.device), targets] # B
    max_v, _ = ls.max(dim=-1, keepdim=True) # B1
    log_sum = (ls - max_v).exp().sum(dim=-1, keepdim=True).log() #B1
    return (log_sum - target_prob[:, None] + max_v).mean()

class MyAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            params = group['params']
            lr = group['lr']
            b1 = group['betas'][0]
            b2 = group['betas'][1]
            eps = group['eps']
            weight_decay = group['weight_decay']
            for param in params:
                if param.grad is None:
                    continue
                g = param.grad
                state = self.state[param]
                if len(state) == 0:
                    state['m'] = torch.zeros_like(param)
                    state['v'] = torch.zeros_like(param)
                    state['t'] = 1
                state['m'] = state['m'] * b1 + (1-b1) * g
                state['v'] = state['v'] * b2 + (1-b2) * g * g
                t = state['t']
                adjusted_lr = lr * (1 - b2 ** t) ** 0.5 / (1-b1**t)
                # print(f'param before={param}, grad={g}')
                param -= adjusted_lr * state['m'] / (state['v']**0.5+eps)
                param -= lr * weight_decay * param
                # print(f'param after={param}')
                state['t'] = t+1

def cos_lr_schedule(lr_max, lr_min, warmup_steps, cos_steps, cur_step):
    if cur_step < warmup_steps:
        return lr_max * cur_step / warmup_steps
    elif cur_step <= cos_steps:
        return lr_min + 0.5 * (1+ math.cos((cur_step-warmup_steps)/(cos_steps-warmup_steps) * math.pi)) * (lr_max - lr_min)
    return lr_min

@torch.no_grad()
def gradient_clipping(parameters:Iterable[torch.nn.Parameter], max_norm:float):
    global_l2 = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        global_l2 += param.grad.square().sum().item()
    global_l2 = math.sqrt(global_l2)
    if global_l2 > max_norm:
        for param in parameters:
            if param.grad is not None:
                param.grad *= max_norm / (global_l2 + 1e-6)

if __name__ == "__main__":
    d_model = 64
    d_ff = 128
    x = torch.randn((2, 4, d_model))
    w1 = torch.randn((d_ff, d_model))
    w2 = torch.randn((d_model, d_ff))
    w3 = torch.randn((d_ff, d_model))
    swiglu = SwiGLU(w1, w2, w3, d_model, d_ff)
    print(swiglu(x).shape)
