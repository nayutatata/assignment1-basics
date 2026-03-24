import torch
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.w = torch.nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.w.T)

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.vocab_size = num_embeddings
        self.d_model = embedding_dim
        self.embedding = torch.nn.Parameter(torch.empty((self.vocab_size, self.d_model), device=device, dtype=dtype))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding[token_ids]
    
class RmsNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.g = torch.nn.Parameter(torch.ones((d_model,), device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        x_var = x.pow(2).mean(-1, keepdim=True) + self.eps
        res = x * x_var.rsqrt() * self.g
        return res.to(in_dtype)

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.ones((d_ff, d_model), dtype=dtype, device=device))
        self.w2 = torch.nn.Parameter(torch.ones((d_model, d_ff), dtype=dtype, device=device))
        self.w3 = torch.nn.Parameter(torch.ones((d_ff, d_model), dtype=dtype, device=device))
        self.d_model = d_model
        self.d_ff = d_ff

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layer1_res = self.silu(x@self.w1.T)
        gate  = x @ self.w3.T
        after_gate = layer1_res * gate
        return after_gate @ self.w2.T

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
        out = torch.empty_like(x)
        out[..., ::2] = x1_cos - x2_sin
        out[..., 1::2] = x1_sin + x2_cos
        return out
    
def my_softmax(x: torch.Tensor, dim:int):
    max_v, _ = x.max(dim=dim, keepdim=True)
    x = (x - max_v).exp()
    x_esum = x.sum(dim=dim, keepdim=True)
    return x/x_esum

def my_scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor)->torch.Tensor:
    q_k = q @ k.transpose(dim0=-1, dim1=-2) / k.shape[-1]**0.5
    attn_mask = torch.where(mask, 0.0, -float('inf'))
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
        self.q_proj = torch.nn.Parameter(torch.ones((d_model, d_model), device=device, dtype=dtype))
        self.k_proj = torch.nn.Parameter(torch.ones((d_model, d_model), device=device, dtype=dtype))
        self.v_proj = torch.nn.Parameter(torch.ones((d_model, d_model), device=device, dtype=dtype))
        self.o_proj = torch.nn.Parameter(torch.ones((d_model, d_model), device=device, dtype=dtype))
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        q = x @ self.q_proj.transpose(-1, -2)
        k = x @ self.k_proj.transpose(-1, -2)
        v = x @ self.v_proj.transpose(-1, -2)
        q = q.reshape(*q.shape[:-1], self.num_heads, self.head_dim).transpose(-3, -2)
        k = k.reshape(*k.shape[:-1], self.num_heads, self.head_dim).transpose(-3, -2)
        v = v.reshape(*v.shape[:-1], self.num_heads, self.head_dim).transpose(-3, -2)
        # here, shape = (batch_size, num_heads, seq_len, head_dim)
        mask = torch.tril(torch.ones((q.shape[-2], q.shape[-2]), dtype=torch.bool))
        atten = my_scaled_dot_product_attention(q,k,v, mask=mask).transpose(-3, -2).reshape(*x.shape)
        return atten @ self.o_proj.transpose(-1, -2)

class MhaWithRope(torch.nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len, theta, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.head_dim = d_model // num_heads
        self.q_proj = torch.nn.Parameter(torch.ones((d_model, d_model), device=device, dtype=dtype))
        self.k_proj = torch.nn.Parameter(torch.ones((d_model, d_model), device=device, dtype=dtype))
        self.v_proj = torch.nn.Parameter(torch.ones((d_model, d_model), device=device, dtype=dtype))
        self.o_proj = torch.nn.Parameter(torch.ones((d_model, d_model), device=device, dtype=dtype))
        self.rope = RoPE(theta=theta, d_k=self.head_dim, max_seq_len=max_seq_len, device=device)

    def forward(self, x:torch.Tensor, token_positions:torch.Tensor) -> torch.Tensor:
        q = x @ self.q_proj.transpose(-1, -2)
        k = x @ self.k_proj.transpose(-1, -2)
        v = x @ self.v_proj.transpose(-1, -2)
        q = q.reshape(*q.shape[:-1], self.num_heads, self.head_dim).transpose(-3, -2)
        k = k.reshape(*k.shape[:-1], self.num_heads, self.head_dim).transpose(-3, -2)
        v = v.reshape(*v.shape[:-1], self.num_heads, self.head_dim).transpose(-3, -2)
        q_rope = self.rope(q, token_positions)
        k_rope = self.rope(k, token_positions)
        mask = torch.tril(torch.ones((q.shape[-2], q.shape[-2]), dtype=torch.bool))
        atten = my_scaled_dot_product_attention(q_rope,k_rope,v, mask=mask).transpose(-3, -2).reshape(*x.shape)
        return atten @ self.o_proj.transpose(-1, -2)

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, d_ff, num_heads, max_seq_len, theta, device=None, dtype=None):
        super().__init__()
        self.attn = MhaWithRope(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.norm1 = RmsNorm(d_model, device=device, dtype=dtype)
        self.norm2 = RmsNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        seq_len = x.shape[-2]
        token_positions = torch.arange(seq_len, device=x.device)
        attn_out = self.attn(x_norm, token_positions)
        x = x + attn_out
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        return x

class TransformerLm(torch.nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float):
        super().__init__()
        self.token_embedding = Embedding(vocab_size, d_model)
        self.layers = torch.nn.ModuleList(
            [
                TransformerBlock(d_model, d_ff, num_heads, context_length, rope_theta)
                for _ in range(num_layers)
            ]
        )
        self.norm = RmsNorm(d_model)
        self.output_embedding = Linear(d_model, vocab_size)
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.output_embedding(x)
        return my_softmax(logits, dim=-1)




if __name__ == "__main__":
    d_model = 64
    d_ff = 128
    x = torch.randn((2, 4, d_model))
    w1 = torch.randn((d_ff, d_model))
    w2 = torch.randn((d_model, d_ff))
    w3 = torch.randn((d_ff, d_model))
    swiglu = SwiGLU(w1, w2, w3, d_model, d_ff)
    print(swiglu(x).shape)
