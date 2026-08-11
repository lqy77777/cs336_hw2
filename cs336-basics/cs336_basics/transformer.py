import torch
import torch.nn as nn
from math import sqrt
from einops import einsum
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

class Linear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            device: torch.device = None,
            dtype: torch.dtype = None,
    ):
        super().__init__()
        self.sigma = sqrt(2 / (out_features + in_features))
        w = torch.empty(out_features,in_features, device = device, dtype = dtype)
        nn.init.trunc_normal_(w, mean = 0.0, std = self.sigma, a = -3 * self.sigma, b = 3 * self.sigma)
        self.weight = nn.Parameter(w)

    def forward(self, x: Tensor) -> Tensor:
        return einsum(self.weight,x,'out d_model, ... d_model -> ... out')

class Embedding(nn.Module):
    def __init__(
            self,
            num_embeddings: int, #size of vocabulary
            embedding_dim: int, #d_model
            device: torch.device = None,
            dtype: torch.dtype = None,
    ):
        super().__init__()
        self.sigma = 1
        w = torch.empty(num_embeddings,embedding_dim, device = device, dtype = dtype)
        nn.init.trunc_normal_(w, mean = 0.0, std = self.sigma, a = -3 * self.sigma, b = 3 * self.sigma)
        self.weight = nn.Parameter(w)
    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(
            self,
            d_model: int,
            eps: float = 1e-5,
            device: torch.device = None,
            dtype: torch.dtype = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device = device, dtype = dtype))
    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32) 
        rms = torch.sqrt(torch.mean(x ** 2, dim = -1, keepdim=True) + self.eps)
        result =  (x / rms) * self.weight
        return result.to(in_dtype)

def SiLU(x):
    return x * torch.sigmoid(x)

class FFN(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            device: torch.device = None,
            dtype: torch.dtype = None,
    ):
        super().__init__()
        self.W_1 = Linear(d_model,d_ff,device,dtype)
        self.W_2 = Linear(d_ff,d_model,device,dtype)
        self.W_3 = Linear(d_model,d_ff,device,dtype)
    def forward(self, x:Tensor) -> Tensor:
        return self.W_2(SiLU(self.W_1(x)) * self.W_3(x))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
            self,
            theta: float,
            d_k: int,  #即d_model
            max_seq_len: int,
            device: torch.device = None
    ):
        super().__init__()
        position = torch.arange(max_seq_len,device = device,dtype = torch.float32)
        k = torch.arange(1,(d_k/2) + 1,device = device)
        denomitor = theta ** (-(2*k - 2)/ d_k)
        #外积
        rope = einsum(position, denomitor, 'i,j -> i j')
        self.register_buffer('rope_cos',torch.cos(rope),persistent = False)
        self.register_buffer('rope_sin',torch.sin(rope),persistent = False)

    def forward(
            self,
            x: Tensor, 
            token_positions: Tensor
    ) -> Tensor:
        cosine = self.rope_cos[token_positions]
        sine = self.rope_sin[token_positions]
        x_1 = x[..., 0::2]
        x_2 = x[..., 1::2]
        y_1 = x_1 * cosine - x_2 * sine
        y_2 = x_1 * sine + x_2 * cosine
        temp = torch.stack([y_1, y_2],dim = -1)
        return rearrange(temp, '... a b -> ... (a b)')

def softmax(x: Tensor, dim: int) -> Tensor:
    shifted = torch.exp(x - torch.max(x, dim = dim, keepdim = True)[0])
    denomitor = torch.sum(shifted,dim = dim,keepdim=True)
    return shifted / denomitor

def scaled_dot_product_attention(
        Q: Float[Tensor,'batch_size ... n d_k'],
        K: Float[Tensor,'batch_size ... m d_k'],
        V: Float[Tensor,'batch_size ... m d_v'],
        mask: Tensor = None
) -> Float[Tensor,'batch_size ... m d_v']:
    d_k= Q.size(-1)
    A = einsum(Q, K, '... n d_k, ... m d_k -> ... n m') / sqrt(d_k)
    if mask is not None:
        A.masked_fill_(~mask, float('-inf'))
    return softmax(A,-1) @ V

class multihead_self_attention(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            theta: float = None,
            max_seq_len: int = None,
            device: torch.device = None,
            dtype: torch.dtype = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.h = num_heads
        self.device = device
        self.theta = theta
        d_k = d_v = d_model // num_heads
        self.W_Q = Linear(d_model, num_heads * d_k,device,dtype)
        self.W_K = Linear(d_model, num_heads * d_k,device,dtype)
        self.W_V = Linear(d_model, num_heads * d_v,device,dtype)
        self.W_O = Linear(num_heads * d_v, d_model,device,dtype)
        if theta is not None:
            self.rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len, device)


    def forward(
            self, 
            x: Tensor, 
            token_positions: Tensor = None
    ) -> Tensor:
        seq_len = x.size(-2)
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        Q = rearrange(Q, '... seq_len (h d_k) -> ... h seq_len d_k',h = self.h)
        K = rearrange(K, '... seq_len (h d_k) -> ... h seq_len d_k',h = self.h)
        V = rearrange(V, '... seq_len (h d_v) -> ... h seq_len d_v',h = self.h)

        if self.theta is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len,device=self.device)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        mask = torch.tril(torch.ones(seq_len,seq_len, device = self.device,dtype = torch.bool))
        A = scaled_dot_product_attention(Q, K, V, mask)
        A = rearrange(A, '... h seq_len d_v -> ... seq_len (h d_v)')
        return self.W_O(A)

class transformer_block(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            theta: float = None,
            max_seq_len: int = None,
            device: torch.device = None,
            dtype: torch.dtype = None,
    ):
        super().__init__()
        self.rms1 = RMSNorm(d_model,device = device, dtype = dtype)
        self.rms2 = RMSNorm(d_model,device = device, dtype = dtype)
        self.attention = multihead_self_attention(d_model,num_heads,theta,max_seq_len,device,dtype)
        self.ffn = FFN(d_model, d_ff, device, dtype)
    def forward(
            self, 
            x: Tensor, 
            token_positions: Tensor = None
    ) -> Tensor:
        y = x + self.attention(self.rms1(x),token_positions)
        return y + self.ffn(self.rms2(y))
class transformer_lm(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            num_layers: int,
            d_model: int,
            num_heads: int,
            d_ff: int,
            rope_theta: float,
            device: torch.device = None,
            dtype: torch.dtype = None,
    ):
        super().__init__()
        self.num_layers= num_layers
        self.embedding = Embedding(vocab_size, d_model,device,dtype)
        self.transformers = nn.ModuleList()
        for i in range(num_layers):
            block = transformer_block(d_model, num_heads, d_ff,rope_theta,
                                      context_length,device,dtype)
            self.transformers.append(block)
        self.rms = RMSNorm(d_model,device=device,dtype = dtype)
        self.lm_head = Linear(d_model, vocab_size,device,dtype)

    def forward(
            self,
            token_ids: Tensor,
    ) -> Tensor:
        x = self.embedding(token_ids)
        for i in range(self.num_layers):
            x = self.transformers[i](x)
        return self.lm_head(self.rms(x))

