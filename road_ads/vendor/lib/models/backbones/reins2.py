import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
from operator import mul
from torch import Tensor
import logging


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
        self.logger = logging.getLogger()

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


class DualLoRAReins(nn.Module):
    """
    双LoRA Reins模块：包含两套独立的LoRAReins参数
    - reins_default: 默认初始化的LoRAReins模块
    - reins_pretrained: 用于加载预训练参数的LoRAReins模块
    - alpha: 可学习的融合系数，范围[0,1]，初始为0表示优先使用预训练特征
    最终特征 = alpha * default_feat + (1 - alpha) * pretrained_feat
    """
    def __init__(
        self,
        num_layers: int,
        embed_dims: int,
        patch_size: int,
        non_adapter_layers: int = 0,
        query_dims: int = 256,
        token_length: int = 100,
        use_softmax: bool = True,
        link_token_to_query: bool = True,
        scale_init: float = 0.001,
        zero_mlp_delta_f: bool = False,
        freeze_pretrained: bool = False,  # 是否冻结预训练的Rein参数
        lora_dim: int = 16,  # LoRA的秩
        alpha_init: float = -5.0,  # alpha的初始化值
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.non_adapter_layers = non_adapter_layers
        self.valid_layers = num_layers - non_adapter_layers
        self.freeze_pretrained = freeze_pretrained
        self.lora_dim = lora_dim
        self.alpha_init = alpha_init
        self.logger = logging.getLogger()
        
        # 创建默认初始化的LoRAReins模块
        self.reins_default = LoRAReins(
            lora_dim=lora_dim,
            num_layers=num_layers,
            embed_dims=embed_dims,
            patch_size=patch_size,
            non_adapter_layers=non_adapter_layers,
            query_dims=query_dims,
            token_length=token_length,
            use_softmax=use_softmax,
            link_token_to_query=link_token_to_query,
            scale_init=scale_init,
            zero_mlp_delta_f=zero_mlp_delta_f,
        )
        
        # 创建预训练的LoRAReins模块（用于加载源域预训练参数）
        self.reins_pretrained = LoRAReins(
            lora_dim=lora_dim,
            num_layers=num_layers,
            embed_dims=embed_dims,
            patch_size=patch_size,
            non_adapter_layers=non_adapter_layers,
            query_dims=query_dims,
            token_length=token_length,
            use_softmax=use_softmax,
            link_token_to_query=link_token_to_query,
            scale_init=scale_init,
            zero_mlp_delta_f=zero_mlp_delta_f,
        )
        
        # 可学习的融合系数alpha，使用配置的初始化值
        # sigmoid(-10) ≈ 0.000045, sigmoid(-5) ≈ 0.0067, sigmoid(0) = 0.5, sigmoid(5) ≈ 0.993
        self.alpha = nn.Parameter(torch.full((self.valid_layers,), alpha_init))
        
        # 如果需要冻结预训练的Rein参数
        if self.freeze_pretrained:
            for param in self.reins_pretrained.parameters():
                param.requires_grad = False
    
    def load_pretrained_reins(self, pretrained_state_dict):
        """
        加载预训练的LoRAReins参数到reins_pretrained模块
        Args:
            pretrained_state_dict: 预训练模型的state_dict
        """
        # 提取Reins相关的参数
        reins_state_dict = {}
        for key, value in pretrained_state_dict.items():
            # 检查是否包含Reins相关的参数名
            reins_param_names = ['learnable_tokens_a', 'learnable_tokens_b', 'scale', 'mlp_token2feat', 'mlp_delta_f', 'transform', 'merge']
            if any(param_name in key for param_name in reins_param_names):
                # 移除可能的前缀（reins., adapter., adapter.reins.等）
                new_key = key
                # 尝试移除各种可能的前缀
                for prefix in ['backbone.adapter.', 'backbone.reins.', 'adapter.', 'reins.']:
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix):]
                        break
                
                # 如果key中还包含中间的reins.，也要移除
                if 'reins.' in new_key:
                    parts = new_key.split('reins.')
                    new_key = parts[-1]  # 取最后一部分
                
                reins_state_dict[new_key] = value
        
        # 加载到reins_pretrained
        missing_keys, unexpected_keys = self.reins_pretrained.load_state_dict(reins_state_dict, strict=False)
        loaded_keys = len(reins_state_dict) - len(unexpected_keys)
        total_keys = len(self.reins_pretrained.state_dict())
        self.logger.info(f"Loaded {loaded_keys}/{total_keys} pretrained LoRAReins parameters.")
        self.logger.info(f"Loaded pretrained LoRAReins parameters: {list(reins_state_dict.keys())}")
        if missing_keys:
            self.logger.warning(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            self.logger.warning(f"Unexpected keys: {unexpected_keys}")
            
        # print(f"Loaded {loaded_keys}/{total_keys} pretrained LoRAReins parameters.")
        # print(f"Loaded pretrained LoRAReins parameters: {list(reins_state_dict.keys())}")
        # if missing_keys:
        #     print(f"Missing keys: {missing_keys}")
        # if unexpected_keys:
        #     print(f"Unexpected keys: {unexpected_keys}")
        
        # 如果设置了冻结，确保参数不可训练
        if self.freeze_pretrained:
            for param in self.reins_pretrained.parameters():
                param.requires_grad = False
    
    def forward(
        self, feats: Tensor, layer: int, batch_first=False, has_cls_token=True, num_register_token=0
    ) -> Tensor:
        """
        前向传播：融合两个LoRAReins模块的输出
        """
        # 验证层索引
        if layer < self.non_adapter_layers or layer >= self.num_layers:
            raise ValueError(f"Layer {layer} is out of valid range [{self.non_adapter_layers}, {self.num_layers-1}]")
        
        # 保存原始输入用于计算delta
        feats_original = feats.clone()
        
        # 通过默认LoRAReins模块
        feats_default = self.reins_default(
            feats.clone(), layer, batch_first, has_cls_token, num_register_token
        )
        
        # 通过预训练LoRAReins模块
        feats_pretrained = self.reins_pretrained(
            feats.clone(), layer, batch_first, has_cls_token, num_register_token
        )
        
        # 计算delta特征（相对于原始输入的变化）
        if batch_first:
            feats_original_perm = feats_original
            feats_default_perm = feats_default
            feats_pretrained_perm = feats_pretrained
        else:
            feats_original_perm = feats_original.permute(1, 0, 2)  # N, B, C -> B, N, C
            feats_default_perm = feats_default.permute(1, 0, 2)
            feats_pretrained_perm = feats_pretrained.permute(1, 0, 2)
        
        delta_default = feats_default_perm - feats_original_perm
        delta_pretrained = feats_pretrained_perm - feats_original_perm
        
        # 获取当前层的alpha值并应用sigmoid确保在[0,1]范围
        adjusted_layer = layer - self.non_adapter_layers
        alpha_value = torch.sigmoid(self.alpha[adjusted_layer])
        
        # 融合特征：alpha * default + (1 - alpha) * pretrained
        delta_fused = alpha_value * delta_default + (1 - alpha_value) * delta_pretrained
        feats_fused = feats_original_perm + delta_fused
        
        # 恢复原始的维度顺序
        if not batch_first:
            feats_fused = feats_fused.permute(1, 0, 2)  # B, N, C -> N, B, C
        
        return feats_fused
    
    def return_auto(self, feats):
        """
        返回自动生成的query（使用默认LoRAReins模块）
        """
        return self.reins_default.return_auto(feats)
    
    def get_alpha_values(self):
        """
        获取当前的alpha融合系数值（经过sigmoid）
        """
        return torch.sigmoid(self.alpha).detach().cpu().numpy()
    
    def set_freeze_pretrained(self, freeze: bool):
        """
        动态设置是否冻结预训练参数
        """
        self.freeze_pretrained = freeze
        for param in self.reins_pretrained.parameters():
            param.requires_grad = not freeze


class DualReins(nn.Module):
    """
    双Reins模块：包含两套独立的Reins参数
    - reins_default: 默认初始化的Reins模块
    - reins_pretrained: 用于加载预训练参数的Reins模块
    - alpha: 可学习的融合系数，范围[0,1]，初始为0表示优先使用预训练特征
    最终特征 = alpha * default_feat + (1 - alpha) * pretrained_feat
    """
    def __init__(
        self,
        num_layers: int,
        embed_dims: int,
        patch_size: int,
        non_adapter_layers: int = 0,
        query_dims: int = 256,
        token_length: int = 100,
        use_softmax: bool = True,
        link_token_to_query: bool = True,
        scale_init: float = 0.001,
        zero_mlp_delta_f: bool = False,
        freeze_pretrained: bool = False,  # 是否冻结预训练的Rein参数
        alpha_init: float = -5.0,  # alpha的初始化值
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.non_adapter_layers = non_adapter_layers
        self.valid_layers = num_layers - non_adapter_layers
        self.freeze_pretrained = freeze_pretrained
        self.alpha_init = alpha_init
        
        # 创建默认初始化的Reins模块
        self.reins_default = Reins(
            num_layers=num_layers,
            embed_dims=embed_dims,
            patch_size=patch_size,
            non_adapter_layers=non_adapter_layers,
            query_dims=query_dims,
            token_length=token_length,
            use_softmax=use_softmax,
            link_token_to_query=link_token_to_query,
            scale_init=scale_init,
            zero_mlp_delta_f=zero_mlp_delta_f,
        )
        
        # 创建预训练的Reins模块（用于加载源域预训练参数）
        self.reins_pretrained = Reins(
            num_layers=num_layers,
            embed_dims=embed_dims,
            patch_size=patch_size,
            non_adapter_layers=non_adapter_layers,
            query_dims=query_dims,
            token_length=token_length,
            use_softmax=use_softmax,
            link_token_to_query=link_token_to_query,
            scale_init=scale_init,
            zero_mlp_delta_f=zero_mlp_delta_f,
        )
        
        # 可学习的融合系数alpha，使用配置的初始化值
        # sigmoid(-10) ≈ 0.000045, sigmoid(-5) ≈ 0.0067, sigmoid(0) = 0.5, sigmoid(5) ≈ 0.993
        self.alpha = nn.Parameter(torch.full((self.valid_layers,), alpha_init))
        
        # 如果需要冻结预训练的Rein参数
        if self.freeze_pretrained:
            for param in self.reins_pretrained.parameters():
                param.requires_grad = False
    
    def load_pretrained_reins(self, pretrained_state_dict):
        """
        加载预训练的Reins参数到reins_pretrained模块
        Args:
            pretrained_state_dict: 预训练模型的state_dict
        """
        # 提取Reins相关的参数
        reins_state_dict = {}
        for key, value in pretrained_state_dict.items():
            # 检查是否包含Reins相关的参数名
            reins_param_names = ['learnable_tokens', 'learnable_tokens_a', 'learnable_tokens_b', 'scale', 'mlp_token2feat', 'mlp_delta_f', 'transform', 'merge']
            if any(param_name in key for param_name in reins_param_names):
                # 移除可能的前缀（reins., adapter., adapter.reins.等）
                new_key = key
                # 尝试移除各种可能的前缀
                for prefix in ['backbone.adapter.', 'backbone.reins.', 'adapter.', 'reins.']:
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix):]
                        break
                
                # 如果key中还包含中间的reins.，也要移除
                if 'reins.' in new_key:
                    parts = new_key.split('reins.')
                    new_key = parts[-1]  # 取最后一部分
                
                reins_state_dict[new_key] = value
        
        # 加载到reins_pretrained
        missing_keys, unexpected_keys = self.reins_pretrained.load_state_dict(reins_state_dict, strict=False)
        print(f"Loaded pretrained Reins parameters: {list(reins_state_dict.keys())}")
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        
        # 如果设置了冻结，确保参数不可训练
        if self.freeze_pretrained:
            for param in self.reins_pretrained.parameters():
                param.requires_grad = False
    
    def forward(
        self, feats: Tensor, layer: int, batch_first=False, has_cls_token=True, num_register_token=0
    ) -> Tensor:
        """
        前向传播：融合两个Reins模块的输出
        """
        # 验证层索引
        if layer < self.non_adapter_layers or layer >= self.num_layers:
            raise ValueError(f"Layer {layer} is out of valid range [{self.non_adapter_layers}, {self.num_layers-1}]")
        
        # 保存原始输入用于计算delta
        feats_original = feats.clone()
        
        # 通过默认Reins模块
        feats_default = self.reins_default(
            feats.clone(), layer, batch_first, has_cls_token, num_register_token
        )
        
        # 通过预训练Reins模块
        feats_pretrained = self.reins_pretrained(
            feats.clone(), layer, batch_first, has_cls_token, num_register_token
        )
        
        # 计算delta特征（相对于原始输入的变化）
        if batch_first:
            feats_original_perm = feats_original
            feats_default_perm = feats_default
            feats_pretrained_perm = feats_pretrained
        else:
            feats_original_perm = feats_original.permute(1, 0, 2)  # N, B, C -> B, N, C
            feats_default_perm = feats_default.permute(1, 0, 2)
            feats_pretrained_perm = feats_pretrained.permute(1, 0, 2)
        
        delta_default = feats_default_perm - feats_original_perm
        delta_pretrained = feats_pretrained_perm - feats_original_perm
        
        # 获取当前层的alpha值并应用sigmoid确保在[0,1]范围
        adjusted_layer = layer - self.non_adapter_layers
        alpha_value = torch.sigmoid(self.alpha[adjusted_layer])
        
        # 融合特征：alpha * default + (1 - alpha) * pretrained
        delta_fused = alpha_value * delta_default + (1 - alpha_value) * delta_pretrained
        feats_fused = feats_original_perm + delta_fused
        
        # 恢复原始的维度顺序
        if not batch_first:
            feats_fused = feats_fused.permute(1, 0, 2)  # B, N, C -> N, B, C
        
        return feats_fused
    
    def return_auto(self, feats):
        """
        返回自动生成的query（使用默认Reins模块）
        """
        return self.reins_default.return_auto(feats)
    
    def get_alpha_values(self):
        """
        获取当前的alpha融合系数值（经过sigmoid）
        """
        return torch.sigmoid(self.alpha).detach().cpu().numpy()
    
    def set_freeze_pretrained(self, freeze: bool):
        """
        动态设置是否冻结预训练参数
        """
        self.freeze_pretrained = freeze
        for param in self.reins_pretrained.parameters():
            param.requires_grad = not freeze
