import torch
import torch.nn as nn
import torch.nn.functional as F


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

# 自定义一个简单的注册器
class PaddingLayerRegistry:
    """Registry for padding layers."""
    def __init__(self):
        self._registry = {}

    def register_module(self, name, module):
        """Register a module with a name."""
        if name in self._registry:
            raise KeyError(f"Module {name} is already registered.")
        self._registry[name] = module

    def get(self, name):
        """Get a module by name."""
        if name not in self._registry:
            raise KeyError(f"Module {name} is not registered.")
        return self._registry[name]

    def __contains__(self, name):
        return name in self._registry
    
# 创建一个全局的注册器实例
PADDING_LAYERS = PaddingLayerRegistry()

# 注册常用的 padding layers
PADDING_LAYERS.register_module('zero', nn.ZeroPad2d)
PADDING_LAYERS.register_module('reflect', nn.ReflectionPad2d)
PADDING_LAYERS.register_module('replicate', nn.ReplicationPad2d)

def build_padding_layer(cfg, *args, **kwargs):
    """Build padding layer.

    Args:
        cfg (None or dict): The padding layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate a padding layer.

    Returns:
        nn.Module: Created padding layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')

    cfg_ = cfg.copy()
    padding_type = cfg_.pop('type')
    if padding_type not in PADDING_LAYERS:
        raise KeyError(f'Unrecognized padding type {padding_type}.')
    else:
        padding_layer = PADDING_LAYERS.get(padding_type)

    layer = padding_layer(*args, **kwargs, **cfg_)

    return layer

# 自定义一个简单的注册器
class ConvLayerRegistry:
    """Registry for convolution layers."""
    def __init__(self):
        self._registry = {}

    def register_module(self, name, module):
        """Register a module with a name."""
        if name in self._registry:
            raise KeyError(f"Module {name} is already registered.")
        self._registry[name] = module

    def get(self, name):
        """Get a module by name."""
        if name not in self._registry:
            raise KeyError(f"Module {name} is not registered.")
        return self._registry[name]

    def __contains__(self, name):
        return name in self._registry
    
# 创建一个全局的注册器实例
CONV_LAYERS = ConvLayerRegistry()

# 注册常用的 convolution layers
CONV_LAYERS.register_module('Conv1d', nn.Conv1d)
CONV_LAYERS.register_module('Conv2d', nn.Conv2d)
CONV_LAYERS.register_module('Conv3d', nn.Conv3d)
CONV_LAYERS.register_module('Conv', nn.Conv2d)

def build_conv_layer(cfg, *args, **kwargs):
    """Build convolution layer.

    Args:
        cfg (None or dict): The conv layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate a conv layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding conv layer.

    Returns:
        nn.Module: Created conv layer.
    """
    if cfg is None:
        cfg_ = dict(type='Conv2d')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in CONV_LAYERS:
        raise KeyError(f'Unrecognized conv type {layer_type}')
    else:
        conv_layer = CONV_LAYERS.get(layer_type)

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer

import inspect
import torch.nn as nn

# 自定义一个简单的注册器
class NormLayerRegistry:
    """Registry for normalization layers."""
    def __init__(self):
        self._registry = {}

    def register_module(self, name, module):
        """Register a module with a name."""
        if name in self._registry:
            raise KeyError(f"Module {name} is already registered.")
        self._registry[name] = module

    def get(self, name):
        """Get a module by name."""
        if name not in self._registry:
            raise KeyError(f"Module {name} is not registered.")
        return self._registry[name]

    def __contains__(self, name):
        return name in self._registry

# 创建一个全局的注册器实例
NORM_LAYERS = NormLayerRegistry()

# 注册常用的 normalization layers
NORM_LAYERS.register_module('BN', nn.BatchNorm2d)
NORM_LAYERS.register_module('BN1d', nn.BatchNorm1d)
NORM_LAYERS.register_module('BN2d', nn.BatchNorm2d)
NORM_LAYERS.register_module('BN3d', nn.BatchNorm3d)
NORM_LAYERS.register_module('GN', nn.GroupNorm)
NORM_LAYERS.register_module('LN', nn.LayerNorm)
NORM_LAYERS.register_module('IN', nn.InstanceNorm2d)
NORM_LAYERS.register_module('IN1d', nn.InstanceNorm1d)
NORM_LAYERS.register_module('IN2d', nn.InstanceNorm2d)
NORM_LAYERS.register_module('IN3d', nn.InstanceNorm3d)

def infer_abbr(class_type):
    """Infer abbreviation from the class name."""
    if not inspect.isclass(class_type):
        raise TypeError(f'class_type must be a type, but got {type(class_type)}')
    if hasattr(class_type, '_abbr_'):
        return class_type._abbr_
    if issubclass(class_type, nn.InstanceNorm2d):  # IN is a subclass of BN
        return 'in'
    elif issubclass(class_type, nn.BatchNorm2d):
        return 'bn'
    elif issubclass(class_type, nn.GroupNorm):
        return 'gn'
    elif issubclass(class_type, nn.LayerNorm):
        return 'ln'
    else:
        class_name = class_type.__name__.lower()
        if 'batch' in class_name:
            return 'bn'
        elif 'group' in class_name:
            return 'gn'
        elif 'layer' in class_name:
            return 'ln'
        elif 'instance' in class_name:
            return 'in'
        else:
            return 'norm_layer'

def build_norm_layer(cfg, num_features, postfix=''):
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.

    Returns:
        (str, nn.Module): The first element is the layer name consisting of
            abbreviation and postfix, e.g., bn1, gn. The second element is the
            created norm layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in NORM_LAYERS:
        raise KeyError(f'Unrecognized norm type {layer_type}')

    norm_layer = NORM_LAYERS.get(layer_type)
    abbr = infer_abbr(norm_layer)

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer

def is_norm(layer, exclude=None):
    """Check if a layer is a normalization layer.

    Args:
        layer (nn.Module): The layer to be checked.
        exclude (type | tuple[type]): Types to be excluded.

    Returns:
        bool: Whether the layer is a norm layer.
    """
    if exclude is not None:
        if not isinstance(exclude, tuple):
            exclude = (exclude, )
        if not all(isinstance(e, type) for e in exclude):
            raise TypeError(
                f'"exclude" must be either None or type or a tuple of types, '
                f'but got {type(exclude)}: {exclude}')

    all_norm_bases = (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm, nn.LayerNorm)
    return isinstance(layer, all_norm_bases) and not isinstance(layer, exclude)


# 自定义一个简单的注册器
class ActivationLayerRegistry:
    """Registry for activation layers."""
    def __init__(self):
        self._registry = {}

    def register_module(self, name=None, module=None):
        """Register a module with a name."""
        if module is None:
            raise ValueError("Module cannot be None.")
        if name is None:
            name = module.__name__
        if name in self._registry:
            raise KeyError(f"Module {name} is already registered.")
        self._registry[name] = module

    def get(self, name):
        """Get a module by name."""
        if name not in self._registry:
            raise KeyError(f"Module {name} is not registered.")
        return self._registry[name]

    def __contains__(self, name):
        return name in self._registry

# 创建一个全局的注册器实例
ACTIVATION_LAYERS = ActivationLayerRegistry()

# 注册常用的激活函数
for module in [
        nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.RReLU, nn.ReLU6, nn.ELU,
        nn.Sigmoid, nn.Tanh
]:
    ACTIVATION_LAYERS.register_module(module=module)

# 自定义 Clamp 激活函数
class Clamp(nn.Module):
    """Clamp activation layer.

    This activation function is to clamp the feature map value within
    :math:`[min, max]`. More details can be found in ``torch.clamp()``.

    Args:
        min (Number | optional): Lower-bound of the range to be clamped to.
            Default to -1.
        max (Number | optional): Upper-bound of the range to be clamped to.
            Default to 1.
    """

    def __init__(self, min=-1., max=1.):
        super(Clamp, self).__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: Clamped tensor.
        """
        return torch.clamp(x, min=self.min, max=self.max)

# 注册 Clamp 激活函数
ACTIVATION_LAYERS.register_module(name='Clamp', module=Clamp)

# 自定义 GELU 激活函数
class GELU(nn.Module):
    r"""Applies the Gaussian Error Linear Units function:

    .. math::
        \text{GELU}(x) = x * \Phi(x)
    where :math:`\Phi(x)` is the Cumulative Distribution Function for
    Gaussian Distribution.

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """

    def forward(self, input):
        return F.gelu(input)

# 注册 GELU 激活函数
ACTIVATION_LAYERS.register_module(name='GELU', module=GELU)

def build_activation_layer(cfg):
    """Build activation layer.

    Args:
        cfg (dict): The activation layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an activation layer.

    Returns:
        nn.Module: Created activation layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')

    cfg_ = cfg.copy()
    layer_type = cfg_.pop('type')
    if layer_type not in ACTIVATION_LAYERS:
        raise KeyError(f'Unrecognized activation type {layer_type}')

    activation_layer = ACTIVATION_LAYERS.get(layer_type)
    return activation_layer(**cfg_)
