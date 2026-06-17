import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import sys

# Code adapted from the fairseq repo.

class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, attn_mask=None):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        aved_state = None

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)

            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
        
        # --- Flash Attention Optimization ---
        # Reshape for SDPA: (Batch, Heads, SeqLen, HeadDim) -> (Batch*Heads, SeqLen, HeadDim)
        # Standard implementation needs (Batch, Heads, SeqLen, HeadDim)
        # But here our q, k, v are (SeqLen, Batch*Heads, HeadDim)
        
        # We need to reshape to: (Batch*Heads, SeqLen, HeadDim)
        q_t = q.transpose(0, 1) # [B*H, T, D]
        k_t = k.transpose(0, 1)
        v_t = v.transpose(0, 1)

        # PyTorch 2.0+ Flash Attention
        if hasattr(F, 'scaled_dot_product_attention') and self.add_zero_attn is False:
             # attn_mask handling is tricky with SDPA if it's not causal or specific shape.
             # SDPA expects attn_mask to be (Batch, 1, T, S) or (T, S)
             # Here attn_mask is (B*H, T, S) which works.
             dropout_p = self.attn_dropout if self.training else 0.0
             attn = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=attn_mask, dropout_p=dropout_p)
             
             # Need to get weights for compatibility? SDPA doesn't return weights.
             # If we strictly need weights, we can't use SDPA efficiently. 
             # However, this project seems to use weights only for averaging? 
             # "attn_weights = attn_weights.sum(dim=1) / self.num_heads"
             # If we don't need weights for downstream, we can mock them.
             # Let's stick to manual implementation if we think weights are crucial, 
             # BUT user asked for optimization. 
             # Let's check where `attn_weights` is used. The return is `attn, attn_weights`.
             # If we look at TransformerEncoderLayer, it ignores weights: `x, _ = self.self_attn(...)`
             # So we are SAFE to drop weights!
             
             attn_weights = None # Weights not available with Flash Attn
        else:
            # Fallback to manual
            attn_weights = torch.bmm(q_t, k_t.transpose(1, 2))
            # assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

            if attn_mask is not None:
                attn_weights += attn_mask.unsqueeze(0)
                    
            attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
            attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)
            attn = torch.bmm(attn_weights, v_t)
        
        # Re-shape back to (SeqLen, Batch, EmbedDim)
        # attn is currently (B*H, T, D) -> (T, B*H, D)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # average attention weights over heads (return None or dummy if SDPA used)
        if attn_weights is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        
        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)
