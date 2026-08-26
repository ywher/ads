# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from argparse import Namespace
import json
import os
import string
from typing import List
import warnings

import torch
from torch import nn
import torch.nn.functional as F

from .adaptor_registry import adaptor_registry, dict_t, state_t

from .adaptor_generic import GenericAdaptor
from .utils import rank_gate


_VERSION_MAP = {
    'siglip2-g-384': 'google/siglip2-giant-opt-patch16-384',
    'siglip2-so400m': 'google/siglip2-so400m-patch16-naflex',
}


def _get_text_projection_size(config, version: str):
    text_config = getattr(config, 'text_config', None)
    projection_size = getattr(text_config, 'projection_size', None)
    if projection_size is not None:
        return projection_size

    if os.path.isdir(version):
        config_path = os.path.join(version, 'config.json')
        if os.path.isfile(config_path):
            with open(config_path, 'r') as f:
                config_json = json.load(f)
            return config_json.get('text_config', {}).get('projection_size')

    return None


def _load_safetensor_tensor(model_dir: str, key: str):
    from safetensors import safe_open

    index_path = os.path.join(model_dir, 'model.safetensors.index.json')
    if os.path.isfile(index_path):
        with open(index_path, 'r') as f:
            weight_map = json.load(f)['weight_map']
        tensor_file = weight_map[key]
    else:
        tensor_file = 'model.safetensors'

    tensor_path = os.path.join(model_dir, tensor_file)
    with safe_open(tensor_path, framework='pt', device='cpu') as f:
        return f.get_tensor(key)


def _restore_siglip2_text_head(model: nn.Module, version: str, projection_size: int):
    head = getattr(getattr(model, 'text_model', None), 'head', None)
    if not isinstance(head, nn.Linear) or head.out_features == projection_size:
        return

    if not os.path.isdir(version):
        warnings.warn(
            'SigLIP2 text projection size differs from this transformers version, '
            'but the checkpoint is not a local directory. Upgrade transformers if loading fails.'
        )
        return

    try:
        weight = _load_safetensor_tensor(version, 'text_model.head.weight')
        bias = _load_safetensor_tensor(version, 'text_model.head.bias')
    except Exception as e:
        warnings.warn(f'Unable to restore SigLIP2 text head from local safetensors: {e}')
        return

    new_head = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
    new_head.to(device=head.weight.device, dtype=head.weight.dtype)
    with torch.no_grad():
        new_head.weight.copy_(weight.to(device=head.weight.device, dtype=head.weight.dtype))
        if bias is not None:
            new_head.bias.copy_(bias.to(device=head.weight.device, dtype=head.weight.dtype))

    model.text_model.head = new_head


def _load_siglip2_model(version: str):
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(version, trust_remote_code=True)
    projection_size = _get_text_projection_size(config, version)
    hidden_size = getattr(getattr(config, 'text_config', None), 'hidden_size', None)
    may_need_head_patch = projection_size is not None and projection_size != hidden_size

    model = AutoModel.from_pretrained(
        version,
        config=config,
        trust_remote_code=True,
        ignore_mismatched_sizes=may_need_head_patch,
    )

    if may_need_head_patch:
        _restore_siglip2_text_head(model, version, projection_size)

    return model


class SigLIP2Adaptor(GenericAdaptor):
    def __init__(self, main_config: Namespace, adaptor_config: dict_t, state: state_t):
        super().__init__(main_config, adaptor_config, state)

        version = adaptor_config['model']
        version = _VERSION_MAP[version]

        from transformers import AutoTokenizer
        with rank_gate():
            model = _load_siglip2_model(version)
            tokenizer = AutoTokenizer.from_pretrained(version, trust_remote_code=True, use_fast=True)

        self.tokenizer = SigLIP2WrappedTokenizer(tokenizer)
        self.text_model = model.text_model

        del model

    def encode_text(self, text, normalize: bool = False):
        output = self.text_model(**text, return_dict=True)
        token = output.pooler_output

        if normalize:
            token = F.normalize(token, dim=-1)

        return token


class SigLIP2WrappedTokenizer:
    def __init__(self, proc):
        self._proc = proc

    def __call__(self, text: List[str]):
        text = [canonicalize_text(t) for t in text]
        ret = self._proc(text=text, return_tensors='pt', max_length=64, padding='max_length', truncation=True)
        return ret


def canonicalize_text(
    text: str,
    *,
    keep_punctuation_exact_string=None,
    trans_punctuation: dict = str.maketrans("", "", string.punctuation),
):
    """Returns canonicalized `text` (lowercase and punctuation removed).

    From: https://github.com/google-research/big_vision/blob/53f18caf27a9419231bbf08d3388b07671616d3d/big_vision/evaluators/proj/image_text/prompt_engineering.py#L94

    Args:
      text: string to be canonicalized.
      keep_punctuation_exact_string: If provided, then this exact string kept.
        For example providing '{}' will keep any occurrences of '{}' (but will
        still remove '{' and '}' that appear separately).
    """
    text = text.replace("_", " ")
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(trans_punctuation)
            for part in text.split(keep_punctuation_exact_string)
        )
    else:
        text = text.translate(trans_punctuation)
    text = text.lower()
    text = " ".join(text.split())
    return text.strip()


@adaptor_registry.register_adaptor("siglip2")
def create_siglip2_adaptor(main_config: Namespace, adaptor_config: dict_t, state: state_t):
    return SigLIP2Adaptor(main_config, adaptor_config, state)
