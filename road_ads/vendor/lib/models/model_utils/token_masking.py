# ---------------------------------------------------------------
# Copyright (c) 2024 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License
# ---------------------------------------------------------------


import torch
from torch import nn


class TokenMasking(nn.Module):
    """
    TokenMasking is a PyTorch module that applies random token masking to the input tensor during training.

    Args:
        mask_token (torch.Tensor): The token to use for masking.

    Methods:
        forward(x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
            Applies random token masking to the input tensor `x` based on the `mask_ratio` during training.
            If not in training mode, returns the input tensor `x` unchanged.

        get_random_token_mask_idx(x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
            Generates a random token mask index based on the `mask_ratio`.
            Args:
                x (torch.Tensor): The input tensor of shape (B, L, C).
                mask_ratio (float): The ratio of tokens to mask.
            Returns:
                torch.Tensor: A boolean tensor of shape (B, L) indicating which tokens to mask.
    """
    def __init__(self, mask_ratio):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.token_mask = None

    def forward(self, shape=(2, 1024)):
        # shape: (B, L)
        if self.training:
            masks = self.get_random_token_mask_idx(shape)
            return masks
            # mask_token = self.mask_token.to(x.dtype)[0]
            # x[masks] = mask_token
            # x = x.contiguous()
            # return x
        else:
            return None

    def get_random_token_mask_idx(self, shape=(2, 1024)):
        '''
        function to generate random token mask index based on mask_ratio
        keep the mask ratio of tokens in the input tensor x
        shape: (B, L)
        '''
        # B: batch size, L: sequence length, C: feature dimension
        B, L = shape
        # generate a random tensor with the same batch size and sequence length as the input tensor x
        token_mask = torch.rand((B, L))   # , device=x.device, (0, 1)
        # generate a boolean tensor based on the mask ratio
        token_mask = self.mask_ratio > token_mask  # True for masking as learnable token
        self.token_mask = token_mask
        return token_mask
