import torch
import torch.nn as nn
from math import sqrt,cos,pi
from einops import einsum
from einops import rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor
from collections.abc import Callable, Iterable
from typing import Optional


def cross_entropy(
        logits: Float[Tensor,'... vocab_size'],  
        target: Int[Tensor,'...']  #第i+1个位置的真实token编号
) -> Tensor:
    #虽然cross_entropy的结果是一个数字，但这个数字需要反向传播
    #所以return的其实是一个0维tensor，不能是float
    max_logits = torch.max(logits,dim = -1, keepdim= True)[0]
    denominator = torch.log(torch.sum(
        torch.exp(logits-max_logits),dim = -1))
    numerator = torch.gather(
        logits, dim = -1,index = target.unsqueeze(-1)).squeeze(-1)
    single = - numerator + max_logits[...,0] + denominator
    return torch.mean(single)

class AdamW(torch.optim.Optimizer):
    def __init__(self, params,
                  lr = 1e-3,
                  betas = (0.9,0.95),
                  eps = 1e-8,
                  weight_decay = 0.1
    ):
        if lr < 0:
            raise ValueError(f'Invalid learning rate: {lr}')
        defaults = {'lr':lr,
                    'betas': betas,
                    'eps': eps,
                    'weight_decay': weight_decay
        }   #例行惯例
        super().__init__(params, defaults)
    @torch.no_grad
    def step(self, closure: Optional[Callable] = None):
        loss = None 
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            beta_1, beta_2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 1) # Get iteration number from the state, or 1.
                if 'm' not in state:
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                m = state['m']
                v = state['v']
                grad = p.grad # Get the gradient of loss with respect to p.
                alpha = lr * (sqrt(1-beta_2 ** t) / (1-beta_1 ** t))
                #原地操作，节省显存
                m.mul_(beta_1).add_(grad, alpha = 1-beta_1)
                v.mul_(beta_2).addcmul_(grad,grad,value=1-beta_2)
                p -= lr * weight_decay * p
                p -= alpha * (m / (torch.sqrt(v) + eps))
                state["t"] = t + 1 # Increment iteration number.
        return loss

def cosine_learning_rate(
        t,
        alpha_max,
        alpha_min,
        T_w,
        T_c
) -> float:
    if t < T_w:
        alpha = (t/T_w) * alpha_max
    elif t <= T_c:
        temp = ((t-T_w)/(T_c-T_w)) * pi
        alpha = alpha_min + ((1+cos(temp)) * (alpha_max-alpha_min)) / 2
    else:
        alpha = alpha_min
    return alpha

def grad_global_norm(parameters: Iterable[nn.Parameter]) -> float:
    """
    """
    total = None
    for p in parameters:
        if p.grad is None:
            continue
        sq = torch.sum(p.grad.detach() ** 2)
        total = sq if total is None else total + sq
    return 0.0 if total is None else sqrt(total.item())

def gradient_clipping(
        p: Iterable[nn.Parameter],
        M: float,
        eps: float =1e-6
):
    p = list(p)
    norm = grad_global_norm(p)
    if norm > M:
        with torch.no_grad():
            for parameter in p:
                if parameter.grad is None:
                    continue
                scale = ( M /( norm + eps))
                parameter.grad *= scale
    return norm
