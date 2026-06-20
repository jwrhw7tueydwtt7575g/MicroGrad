"""2D Convolutional layer."""
import numpy as np
from .. import _C
from .module import Module
from ..tensor import tensor


class Conv2d(Module):
    """2D Convolution layer.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    kernel_size : int or tuple
    stride : int (default 1)
    padding : int (default 0)
    bias : bool (default True)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if isinstance(kernel_size, (list, tuple)):
            if len(kernel_size) == 2 and kernel_size[0] != kernel_size[1]:
                raise ValueError("Conv2d only supports square kernels in v0.1")
            self.kernel_size = kernel_size[0]
        else:
            self.kernel_size = kernel_size
            
        self.stride = stride
        self.padding = padding
        self.has_bias = bias

        # Kaiming uniform init.
        fan_in = in_channels * self.kernel_size * self.kernel_size
        bound = (1.0 / fan_in) ** 0.5
        rng = np.random.default_rng(42)
        
        w_shape = [out_channels, in_channels, self.kernel_size, self.kernel_size]
        w_data = rng.uniform(-bound, bound, size=w_shape).astype(np.float32)
        w = _C.Tensor(w_data.reshape(-1).tolist(), w_shape)
        w.requires_grad_(True)
        self.add_param("weight", w)

        if bias:
            b_data = [0.0] * out_channels
            b = _C.Tensor(b_data, [out_channels])
            b.requires_grad_(True)
            self.add_param("bias", b)
        else:
            b = _C.Tensor([0.0] * out_channels, [out_channels])
            b.requires_grad_(False)
            self.add_param("bias", b)
            
        self.stride_tensor = tensor([float(stride)])
        self.padding_tensor = tensor([float(padding)])

    def forward(self, x):
        w = self._params[0][1]
        b = self._params[1][1]
        return x.conv2d(w, b, self.stride_tensor, self.padding_tensor)
