import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
from operator import mul
from torch import Tensor


class Reins(nn.Module):
    def __init__(
        self,
        num_layers: int,  # 24
        embed_dims: int,  # 1024
        patch_size: int,  # 16
        non_adapter_layers: int= 0,
        query_dims: int = 256,
        token_length: int = 100,
        use_softmax: bool = True,
        link_token_to_query: bool = True,
        scale_init: float = 0.001,
        zero_mlp_delta_f: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers  # 24
        self.non_adapter_layers = non_adapter_layers  # 8
        self.valid_layers = num_layers - non_adapter_layers  # 16
        self.embed_dims = embed_dims  # 1024    
        self.patch_size = patch_size  # 16
        self.query_dims = query_dims  # 256
        self.token_length = token_length  # 100
        self.link_token_to_query = link_token_to_query  # True
        self.scale_init = scale_init  # 0.001
        self.use_softmax = use_softmax  # True
        self.zero_mlp_delta_f = zero_mlp_delta_f  # False
        self.create_model()

    def create_model(self):
        self.learnable_tokens = nn.Parameter(
            torch.empty([self.valid_layers, self.token_length, self.embed_dims])
        )  # N, m, c
        self.scale = nn.Parameter(torch.tensor(self.scale_init))
        self.mlp_token2feat = nn.Linear(self.embed_dims, self.embed_dims)  # c, c
        self.mlp_delta_f = nn.Linear(self.embed_dims, self.embed_dims)  # c, c
        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1) + self.embed_dims
            )
        )
        nn.init.uniform_(self.learnable_tokens.data, -val, val)
        nn.init.kaiming_uniform_(self.mlp_delta_f.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_token2feat.weight, a=math.sqrt(5))
        # link token to query
        if self.link_token_to_query:
            self.transform = nn.Linear(self.embed_dims, self.query_dims)  # c, q
            self.merge = nn.Linear(self.query_dims * 3, self.query_dims)  # 3q, q
        if self.zero_mlp_delta_f:
            # 不要删除 self.scale，而是重新初始化为固定值
            with torch.no_grad():
                self.scale.fill_(1.0)
            self.scale.requires_grad = False  # 设置为不可训练
            nn.init.zeros_(self.mlp_delta_f.weight)
            nn.init.zeros_(self.mlp_delta_f.bias)
            
    def return_auto(self, feats):
        if self.link_token_to_query:
            tokens = self.transform(self.get_tokens(-1)).permute(1, 2, 0)
            tokens = torch.cat(
                [
                    F.max_pool1d(tokens, kernel_size=self.valid_layers),
                    F.avg_pool1d(tokens, kernel_size=self.valid_layers),
                    tokens[:, :, -1].unsqueeze(-1),
                ],
                dim=-1,
            )
            querys = self.merge(tokens.flatten(-2, -1))
            return feats, querys
        else:
            return feats

    def get_tokens(self, layer: int) -> Tensor:
        if layer == -1:
            # return all
            return self.learnable_tokens
        else:
            # 确保层索引在有效范围内
            adjusted_layer = layer - self.non_adapter_layers
            if adjusted_layer < 0 or adjusted_layer >= self.valid_layers:
                raise IndexError(f"Layer {layer} is not valid. Valid range: [{self.non_adapter_layers}, {self.num_layers-1}]")
            return self.learnable_tokens[adjusted_layer]  # [m, c]

    def forward(
        self, feats: Tensor, layer: int, batch_first=False, has_cls_token=True, num_register_token=0
    ) -> Tensor:
        # 输入验证
        if layer < self.non_adapter_layers or layer >= self.num_layers:
            raise ValueError(f"Layer {layer} is out of valid range [{self.non_adapter_layers}, {self.num_layers-1}]")
        
        if batch_first:
            feats = feats.permute(1, 0, 2)  # B, N, C to N, B, C
        
        # 分离 cls_token
        if has_cls_token:
            if feats.size(0) < 1:
                raise ValueError("Input features must have at least 1 token when has_cls_token=True")
            cls_token, feats = torch.tensor_split(feats, [1], dim=0)  # feats: [N, B, C] 
        
        # 分离 register_token
        if num_register_token > 0:
            if feats.size(0) < num_register_token:
                raise ValueError(f"Input features must have at least {num_register_token} tokens when num_register_token={num_register_token}")
            register_token, feats = torch.tensor_split(feats, [num_register_token], dim=0)  # feats: [N-num_register_token, B, C]
        
        # print(f'the shape of cls token, register token and feats: {cls_token.shape}, {register_token.shape if num_register_token > 0 else None}, {feats.shape}')
        tokens = self.get_tokens(layer)  # m, c
        delta_feat = self.forward_delta_feat(
            feats,
            tokens,
            layer,
        )  # [n, b, c]
        delta_feat = delta_feat * self.scale
        feats = feats + delta_feat  # [n, b, c]
        
        # 重新组合tokens
        if num_register_token > 0:
            feats = torch.cat([register_token, feats], dim=0)
        if has_cls_token:
            feats = torch.cat([cls_token, feats], dim=0)
        
        if batch_first:
            feats = feats.permute(1, 0, 2)  # N, B, C to B, N, C
        return feats

    def forward_delta_feat(self, feats: Tensor, tokens: Tensor, layers: int) -> Tensor:
        # 注意：这里的tokens应该是[m, c]的形状（单层token）
        attn = torch.einsum("nbc,mc->nbm", feats, tokens)  # [n,b,c] @ [m,c] -> [n,b,m]
        if self.use_softmax:
            attn = attn * (self.embed_dims**-0.5)
            attn = F.softmax(attn, dim=-1)  # [nbm]
        
        # 跳过第一个token（通常是特殊token），只使用剩余的tokens进行特征转换
        if tokens.size(0) > 1:
            delta_f = torch.einsum(
                "nbm,mc->nbc",
                attn[:, :, 1:],  # [n,b,m-1] - 跳过第一个token的注意力权重
                self.mlp_token2feat(tokens[1:, :]),  # [m-1,c] - 跳过第一个token
            )  # [n,b,c]
        else:
            # 如果只有一个token，直接使用全部
            delta_f = torch.einsum(
                "nbm,mc->nbc",
                attn,  # [n,b,m]
                self.mlp_token2feat(tokens),  # [m,c]
            )  # [n,b,c]
        
        delta_f = self.mlp_delta_f(delta_f + feats)  # [n,b,c]
        return delta_f


class LoRAReins(Reins):
    def __init__(self, lora_dim=16, **kwargs):
        self.lora_dim = lora_dim
        super().__init__(**kwargs)

    def create_model(self):
        super().create_model()
        del self.learnable_tokens
        self.learnable_tokens_a = nn.Parameter(
            torch.empty([self.valid_layers, self.token_length, self.lora_dim])  # N, m, r
        )
        self.learnable_tokens_b = nn.Parameter(
            torch.empty([self.valid_layers, self.lora_dim, self.embed_dims])  # N, r, c
        )
        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1)
                + (self.embed_dims * self.lora_dim) ** 0.5
            )
        )
        nn.init.uniform_(self.learnable_tokens_a.data, -val, val)
        nn.init.uniform_(self.learnable_tokens_b.data, -val, val)

    def get_tokens(self, layer):
        if layer == -1:
            return self.learnable_tokens_a @ self.learnable_tokens_b
        else:
            # 确保层索引在有效范围内
            adjusted_layer = layer - self.non_adapter_layers
            if adjusted_layer < 0 or adjusted_layer >= self.valid_layers:
                raise IndexError(f"Layer {layer} is not valid. Valid range: [{self.non_adapter_layers}, {self.num_layers-1}]")
            return self.learnable_tokens_a[adjusted_layer] @ self.learnable_tokens_b[adjusted_layer]  # [m, c]
