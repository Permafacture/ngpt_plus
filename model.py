# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# MIT license
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# The text below is the original header from the nanoGPT library
"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func


def apply_rotary_position_embeddings(sinusoidal_pos, q, k):
    # Split the sinusoidal_pos into sin and cos parts
    sin, cos = sinusoidal_pos.chunk(2, dim=-1)
    # Apply the rotary embeddings to the query and key
    q_rot = torch.stack((-q[..., 1::2], q[..., ::2]), dim=-1)
    k_rot = torch.stack((-k[..., 1::2], k[..., ::2]), dim=-1)
    q_rot = torch.reshape(q_rot, q.shape[:-1] + (q.shape[-1]//2, 2)) * torch.stack((cos, sin), dim=-1)
    k_rot = torch.reshape(k_rot, k.shape[:-1] + (k.shape[-1]//2, 2)) * torch.stack((cos, sin), dim=-1)
    q_rot = torch.reshape(q_rot, q.shape)
    k_rot = torch.reshape(k_rot, k.shape)
    return q_rot, k_rot

def get_sinusoidal_embeddings( n_positions, dim):
    """Generate sinusoidal positional embeddings."""
    position = torch.arange(n_positions, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    sinusoidal_emb = torch.zeros((n_positions, dim))
    sinusoidal_emb[:, 0::2] = torch.sin(position * div_term)
    sinusoidal_emb[:, 1::2] = torch.cos(position * div_term)
    return sinusoidal_emb

def build_attention(config):
    gqa_embed = config.n_embd * config.n_head_k // config.n_head
    attn = nn.ModuleDict({
        'query': nn.Linear(config.n_embd, config.n_embd, bias=config.bias, dtype=torch.bfloat16),
        'key': nn.Linear(config.n_embd, gqa_embed, bias=config.bias, dtype=torch.bfloat16),
        'value': nn.Linear(config.n_embd, gqa_embed, bias=config.bias, dtype=torch.bfloat16),
        'att_c_proj': nn.Linear(config.n_embd, config.n_embd, bias=config.bias, dtype=torch.bfloat16)
        })
    return attn

class Block(nn.Module):
    def __init__(self, config, iblock, shared_attn=None):
        super().__init__()
        self.config = config
        self.shared_attn = shared_attn

        # Only create attention parameters if not shared
        if not config.share_attention_params:
            attn = build_attention(config)
            self.query = attn.query
            self.key = attn.key
            self.value = attn.value
            self.att_c_proj = attn.att_c_proj
        else:
            self.query = self.shared_attn.query
            self.key = self.shared_attn.key
            self.value = self.shared_attn.value
            self.att_c_proj = self.shared_attn.att_c_proj


        # FFN parameters are never shared
        self.c_fc = nn.Linear(config.n_embd, 2 * 4 * config.n_embd, bias=config.bias, dtype=torch.bfloat16)
        self.silu = nn.SiLU()
        self.mlp_c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias, dtype=torch.bfloat16)

        if (config.use_nGPT == 0):
            self.rmsnorm_att = RMSNorm(config.n_embd)
            self.rmsnorm_mlp = RMSNorm(config.n_embd)

        if (config.use_nGPT == 1):
            self.attn_alpha_init_value = 0.05
            self.attn_alpha_init_scaling = config.base_scale
            self.attn_alpha = torch.nn.Parameter(self.attn_alpha_init_scaling*torch.ones(self.config.n_embd, dtype=torch.float32))

            self.mlp_alpha_init_value = 0.05
            self.mlp_alpha_init_scaling = config.base_scale
            self.mlp_alpha = torch.nn.Parameter(self.mlp_alpha_init_scaling*torch.ones(self.config.n_embd, dtype=torch.float32))

            self.sqk_init_value = 1.0       
            self.sqk_init_scaling = config.base_scale
            self.sqk_q = torch.nn.Parameter(self.sqk_init_scaling*torch.ones(config.n_embd, dtype=torch.float32))
            
            gqa_embed = config.n_embd * config.n_head_k // config.n_head
            self.sqk_k = torch.nn.Parameter(self.sqk_init_scaling*torch.ones(gqa_embed, dtype=torch.float32))

            self.suv_init_value = 1.0
            self.suv_init_scaling = 1.0
            self.suv = torch.nn.Parameter(self.suv_init_scaling*torch.ones(2 * 4 * config.n_embd, dtype=torch.float32))

    
    def justnorm(self, x):
        #return F.normalize(x, p=2, dim=-1)
        res = x / x.norm(p=2, dim=-1, keepdim=True)
        return res

    def forward(self, h):
        B, T, C = h.size()

        hin = h
        if (self.config.use_nGPT == 0):
            hin = self.rmsnorm_att(h)
        
        # Get Q,K,V
        q = self.query(hin)
        k = self.key(hin)
        v = self.value(hin)

        # Apply rotary embeddings before reshaping
        gqa_embed = self.config.n_embd * self.config.n_head_k // self.config.n_head
        q = q.view(B, T, self.config.n_head, self.config.n_embd // self.config.n_head)
        k = k.view(B, T, self.config.n_head_k, gqa_embed // self.config.n_head_k)
        v = v.view(B, T, self.config.n_head_k, gqa_embed // self.config.n_head_k)

        # Generate appropriate sinusoidal embeddings for each
        q_sin_pos = get_sinusoidal_embeddings(T, self.config.n_embd // self.config.n_head).to(device=q.device)
        k_sin_pos = get_sinusoidal_embeddings(T, gqa_embed // self.config.n_head_k).to(device=k.device)
        
        # Apply rotary embeddings separately
        q, _ = apply_rotary_position_embeddings(q_sin_pos, q.transpose(1, 2), q.transpose(1, 2))
        _, k = apply_rotary_position_embeddings(k_sin_pos, k.transpose(1, 2), k.transpose(1, 2))
        q = q.transpose(2, 1)
        k = k.transpose(2, 1)

        if (self.config.use_nGPT == 1):
            # the paper says this may not even be necessary, and I don't know if this is the 
            # right way to handle this with GQA anyway
            sqk_q = (self.sqk_q * (self.sqk_init_value/self.sqk_init_scaling)).view(1, 1, self.config.n_head, self.config.n_embd // self.config.n_head)
            q = sqk_q * self.justnorm(q)

            sqk_k = (self.sqk_k * (self.sqk_init_value/self.sqk_init_scaling)).view(1, 1, self.config.n_head_k, gqa_embed // self.config.n_head_k)
            k = sqk_k * self.justnorm(k)

        sqrt_head_dim = (self.config.n_embd / self.config.n_head) ** 0.5
        if (self.config.use_nGPT == 0): softmax_scale = 1.0 / sqrt_head_dim 
        if (self.config.use_nGPT == 1): softmax_scale = sqrt_head_dim 

        # Update flash attention call to use local window if configured
        window_size = (self.config.local_attention_window, 0) if self.config.local_attention_window > 0 else (-1, -1)
        
        y = flash_attn_func(
            q.to(dtype=torch.bfloat16), 
            k.to(dtype=torch.bfloat16), 
            v.to(dtype=torch.bfloat16), 
            dropout_p=0.0, 
            softmax_scale=softmax_scale, 
            causal=True, 
            window_size=window_size,
            alibi_slopes=None, 
            deterministic=True
        )
        y = y.to(dtype=q.dtype)
        y = y.contiguous().view(B, T, self.config.n_embd)

        h_att = self.att_c_proj(y)

        if (self.config.use_nGPT == 0):
            h = h + h_att
        if (self.config.use_nGPT == 1):
            lr = self.attn_alpha * (self.attn_alpha_init_value / self.attn_alpha_init_scaling)
            lr = torch.abs(lr)
            
            A_norm = self.justnorm(h) # normally, normalization is not needed
            B_norm = self.justnorm(h_att)
                
            #res = (1.0 - lr) * A_norm + lr * B_norm
            res = A_norm + lr * (B_norm - A_norm)
            h = self.justnorm(res)

        hin = h
        if (self.config.use_nGPT == 0):
            hin = self.rmsnorm_mlp(h)
        uv = self.c_fc(hin)
        if (self.config.use_nGPT == 1):
            suv = (self.suv * ((self.suv_init_value/self.suv_init_scaling) * (self.config.n_embd ** 0.5))) 
            uv = suv * uv  
        u, v = torch.chunk(uv, 2, dim=-1)
        x_mlp = u * self.silu(v)
        h_mlp = self.mlp_c_proj(x_mlp)

        if (self.config.use_nGPT == 0):
            h = h + h_mlp
        if (self.config.use_nGPT == 1):
            lr = self.mlp_alpha * (self.mlp_alpha_init_value / self.mlp_alpha_init_scaling)
            lr = torch.abs(lr)

            A_norm = self.justnorm(h) # normally, normalization is not needed
            B_norm = self.justnorm(h_mlp)
                
            #res = (1.0 - lr) * A_norm + lr * B_norm
            res = A_norm + lr * (B_norm - A_norm)
            h = self.justnorm(res)

        return h

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 1024
    base_scale: float = 1.0 / (1024.0 ** 0.5)    # 1 / sqrt(n_embd)
    use_nGPT: int = 0
    dropout: float = 0.0
    bias: bool = False 

    # New parameters
    n_head_k: int = 0  # 0 means standard attention, otherwise number of K/V groups
    local_attention_window: int = -1  # -1 means full attention
    share_attention_params: bool = False
    weight_tying: bool = False

class RMSNorm(torch.nn.Module):
    def __init__(self, embdim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(embdim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = torch.mean(x * x, dim=-1, keepdim=True)
        xnorm = x * torch.rsqrt(norm + self.eps)
        xnorm = xnorm.to(dtype=dtype)
        return xnorm * self.weight


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None

        # validate GQA params
        nhk = config.n_head_k
        nh = config.n_head
        if nhk <= 0:
            nhk = nh
        else:
            assert nhk <= nh
            assert nh % nhk == 0

        self.config = config
    
        transformer_dict = {
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'drop': nn.Dropout(config.dropout),
        }
    
        # Add shared attention parameters if enabled
        shared_attn = None
        if config.share_attention_params:
            shared_attn = build_attention(config)
            transformer_dict['shared_attn'] = shared_attn

        # Add blocks with shared attention if enabled
        transformer_dict['h'] = nn.ModuleList([Block(config, il, shared_attn) for il in range(config.n_layer)])
        
        self.transformer = nn.ModuleDict(transformer_dict)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # *we don't use it becuase in the nGPT paper there was no weight tying of weights*
        if config.weight_tying:
            # https://paperswithcode.com/method/weight-tying
            self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=config.base_scale/math.sqrt(2 * config.n_layer))
        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
    
        if (config.use_nGPT == 1):
            self.sz_init_value = 1.00
            self.sz_init_scaling = config.base_scale
            self.sz = torch.nn.Parameter(self.sz_init_scaling*torch.ones(config.vocab_size, dtype=torch.float32))

        if (config.use_nGPT == 0):
            self.rmsnorm_f = RMSNorm(config.n_embd)


    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        #if non_embedding:
        #    n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_scale)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_scale)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        #assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        
        x = tok_emb
        for block in self.transformer.h:
            x = block(x)

        if (self.config.use_nGPT == 0):
            x = self.rmsnorm_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            if (self.config.use_nGPT == 1):
                sz = self.sz * (self.sz_init_value/self.sz_init_scaling)
                logits = sz * logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            if (self.config.use_nGPT == 1):
                sz = self.sz * (self.sz_init_value/self.sz_init_scaling)
                logits = sz * logits
            loss = None

        return logits, loss


    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = False#fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
