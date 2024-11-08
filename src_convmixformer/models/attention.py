import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from collections import OrderedDict
import torch.nn.functional as F

def position_embedding(input, d_model):
    input = input.view(-1, 1)
    dim = torch.arange(d_model // 2, dtype=torch.float32, device=input.device).view(1, -1)
    sin = torch.sin(input / 10000 ** (2 * dim / d_model))
    cos = torch.cos(input / 10000 ** (2 * dim / d_model))

    out = torch.zeros((input.shape[0], d_model), device=input.device)
    out[:, ::2] = sin
    out[:, 1::2] = cos
    return out

def sinusoid_encoding_table(max_len, d_model):
    pos = torch.arange(max_len, dtype=torch.float32)
    out = position_embedding(pos, d_model)
    return out

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention layer with Dropout and Layer Normalization.
    """

    def __init__(self, d_model,  dff=2048, dropout=.1):
        super(MultiHeadAttention, self).__init__()

        # self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        self.conv_proj_q = self._build_projection(d_model, kernel_size=3, padding=1, stride=1)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        # self.layer_norm2 = nn.LayerNorm(d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        hidden_features = int(d_model)

        self.project_in = nn.Conv1d(d_model, hidden_features, kernel_size=1)

        self.dwconv = nn.Conv1d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features)

        self.project_out = nn.Conv1d(256, d_model, kernel_size=1)


        # self.fc = nn.Sequential(*[nn.Linear(d_model, dff), nn.ReLU(inplace=True), nn.Dropout(p=dropout),
        #                           nn.Linear(dff, d_model)])

    def forward_conv(self, x):
        # if self.with_cls_token:
        #     cls_token, x = torch.split(x, [1, h*w], 1)
        b, f, c = x.shape       

        x = rearrange(x, 'b f c -> b c f')

        q = self.conv_proj_q(x)

        self.cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        # q = torch.cat((cls_tokens, q), dim=1) 
        # # if self.with_cls_token:
        # print(self.cls_tokens.shape)
        # print(q.shape)
        # q = torch.cat((self.cls_tokens, q), dim=1)

        return q
    
    def _build_projection(self,dim_in, kernel_size, padding,stride):
        proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv1d(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=dim_in
                )),
                ('bn', nn.BatchNorm1d(dim_in)),
                ('rearrage', Rearrange('b c f -> b f c')),
            ]))
      
        return proj
    
    def forward(self, queries):
        # att,queries = self.attention(queries, keys, values)
        att = self.forward_conv(queries)

        att = self.dropout(att)
        # att = self.layer_norm(queries + att)
        g = self.project_in(att.permute(0, 2, 1))
        x1, x2 = self.dwconv(g).chunk(2, dim=1)
        g = F.gelu(x1) * x2
        g = self.project_out(g)
        att = self.dropout(g.permute(0, 2, 1))
        return self.layer_norm(queries + att)

class EncoderSelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_head, dff=2048, dropout_transformer=.1, n_module=6):
        super(EncoderSelfAttention, self).__init__()
        # dim = [512, 256, 128, 64 , 32, 16]
        # self.pos_embed = nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model)
        self.encoder = nn.ModuleList([MultiHeadAttention(d_model,  dff, dropout_transformer)
                                      for i in range(n_module)])
    def forward(self, x):
        # print(sinusoid_encoding_table(x.shape[1], x.shape[2]).shape)      #40,512 
        # print(sinusoid_encoding_table(x.shape[1], x.shape[2]).expand(x.shape).shape)          #8,40,5124
        in_encoder = x + sinusoid_encoding_table(x.shape[1], x.shape[2]).expand(x.shape).cuda(device= 0)
        # in_encoder = x + self.pos_embed(x.permute(0,2,1)).permute(0,2,1)
        for l in self.encoder:
            in_encoder = l(in_encoder)
        return in_encoder