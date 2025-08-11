# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus", 
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
    "Index",
    "FeedbackConvLSTMBlock",
    "TupleToTensor",
    "SDM",
    "FusionBlock",
    "Silences",
    "SilenceChannel",
    "ConcatSpe",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Silence(nn.Module):
    def __init__(self):
        super(Silence, self).__init__()
    def forward(self, x):
        return x

class SilenceChannel(nn.Module):
    def __init__(self,c_start, c_end):
        super(SilenceChannel, self).__init__()
        self.c_start=c_start
        self.c_end = c_end
    def forward(self, x):
        newX = x[:, self.c_start:self.c_end, :, :]
        return newX

class SDM(torch.nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        # c1 is a list, e.g., [64, 64]
        # c2 is the scaled output channels, e.g., 64

        self.C = float(c2)  # Use the scaled output channels

        # The input to theta is the sum of the original input channels
        total_input_channels = sum(c1)
        self.theta = torch.nn.Conv2d(in_channels=total_input_channels,
                                     out_channels=c2,  # The output is the scaled value
                                     kernel_size=1)
    def forward(self,x: List[torch.Tensor]):


        fm = x[0]
        ft = x[1]
        B, C, H, W = fm.shape
        B1,C1,H1,W1 = ft.shape
        

        
        # Reshape inputs for matrix multiplication, as described in the paper.
        # Shape: (B, C, H, W) -> (B, C, HW)
        fm_reshaped = fm.view(C, B * H * W)
        ft_reshaped = ft.view(C1, B1 * H1 * W1)
        # Permute to get channels as the last dimension.
        # Shape: (B, C, HW) -> (B, HW, C)
        fm_permuted = fm_reshaped.permute(1, 0)

        # Calculate the similarity matrix S using matrix multiplication.
        # (B, HW, C) @ (B, C, HW) -> (B, HW, HW)
        S = torch.matmul(fm_permuted, ft_reshaped) / torch.sqrt(torch.tensor(self.C))
        # Calculate the attention weights 'w' using softmax.
        # Softmax is applied on the last dimension of S to get weights for each memory pixel.
        w = torch.nn.functional.softmax(S, dim=-1)
        # Apply the attention weights 'w' to the memory features 'fm'.
        # (B, C, HW) @ (B, HW, HW) -> (B, C, HW)
        fm_weighted = torch.matmul(fm_reshaped, w)
        # Reshape the weighted memory features back to image format.
        # Shape: (B, C, HW) -> (B, C, H, W)
        fm_attended = fm_weighted.view(B1, C1, H1, W1)
        # Concatenate the attended memory features and the current features
        # along the channel dimension (dim=1).
        concatenated_features = torch.cat([fm_attended, ft], dim=1)
        # Pass through the final 1x1 convolution (θ).
        output = self.theta(concatenated_features)
        # --- Check the output before returning ---
        return output

class FusionBlock(torch.nn.Module):
    """Encapsulates your 'fusion' logic."""

    def __init__(self, c1, c2, arg_from_yaml):
        super().__init__()
        # c1: The list of input channels, e.g. [64, 64]
        # c2: The scaled output channels for this block, e.g., 128
        # arg_from_yaml: The second argument from your YAML, e.g., 512

        # Now, use these correct variables to build your layers.
        # For example, if your FusionBlock upsamples the first input
        # to match the second, then runs an SDM:

        input1_ch, input2_ch = c1

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.channel_changer = nn.Conv2d(input1_ch, input2_ch, kernel_size=1)

        # The SDM's input channels are [input2_ch, input2_ch]
        # Its output channels should be the final 'c2' for this FusionBlock.
        self.sdm = SDM(c1=[input2_ch, input2_ch], c2=c2)

    def forward(self, x: List[torch.Tensor]):


        f1 = x[0]
        f2 = x[1]
        f1_upsampled = self.upsample(f1)
        f1_ready = self.channel_changer(f1_upsampled)

        return self.sdm([f1_ready, f2])

class Conv(nn.Module):
    """
    Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """
        Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))


class Conv2(Conv):
    """
    Simplified RepConv module with Conv fusing.

    Attributes:
        conv (nn.Conv2d): Main 3x3 convolutional layer.
        cv2 (nn.Conv2d): Additional 1x1 convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv2 layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """
        Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """
        Apply fused convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution module with 1x1 and depthwise convolutions.

    This implementation is based on the PaddleDetection HGNetV2 backbone.

    Attributes:
        conv1 (Conv): 1x1 convolution layer.
        conv2 (DWConv): Depthwise convolution layer.
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """
        Initialize LightConv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for depthwise convolution.
            act (nn.Module): Activation function.
        """
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """
        Apply 2 convolutions to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution module."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """
        Initialize depth-wise convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution module."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """
        Initialize depth-wise transpose convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p1 (int): Padding.
            p2 (int): Output padding.
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """
    Convolution transpose module with optional batch normalization and activation.

    Attributes:
        conv_transpose (nn.ConvTranspose2d): Transposed convolution layer.
        bn (nn.BatchNorm2d | nn.Identity): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """
        Initialize ConvTranspose layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            bn (bool): Use batch normalization.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Apply transposed convolution, batch normalization and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """
        Apply activation and convolution transpose operation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """
    Focus module for concentrating feature information.

    Slices input tensor into 4 parts and concatenates them in the channel dimension.

    Attributes:
        conv (Conv): Convolution layer.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """
        Initialize Focus module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Apply Focus operation and convolution to input tensor.

        Input shape is (B, C, W, H) and output shape is (B, 4C, W/2, H/2).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """
    Ghost Convolution module.

    Generates more features with fewer parameters by using cheap operations.

    Attributes:
        cv1 (Conv): Primary convolution.
        cv2 (Conv): Cheap operation convolution.

    References:
        https://github.com/huawei-noah/Efficient-AI-Backbones
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """
        Initialize Ghost Convolution module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """
        Apply Ghost Convolution to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor with concatenated features.
        """
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv module with training and deploy modes.

    This module is used in RT-DETR and can fuse convolutions during inference for efficiency.

    Attributes:
        conv1 (Conv): 3x3 convolution.
        conv2 (Conv): 1x1 convolution.
        bn (nn.BatchNorm2d, optional): Batch normalization for identity branch.
        act (nn.Module): Activation function.
        default_act (nn.Module): Default activation function (SiLU).

    References:
        https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """
        Initialize RepConv module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
            bn (bool): Use batch normalization for identity branch.
            deploy (bool): Deploy mode for inference.
        """
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """
        Forward pass for deploy mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))

    def forward(self, x):
        """
        Forward pass for training mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """
        Calculate equivalent kernel and bias by fusing convolutions.

        Returns:
            (torch.Tensor): Equivalent kernel
            (torch.Tensor): Equivalent bias
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """
        Pad a 1x1 kernel to 3x3 size.

        Args:
            kernel1x1 (torch.Tensor): 1x1 convolution kernel.

        Returns:
            (torch.Tensor): Padded 3x3 kernel.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """
        Fuse batch normalization with convolution weights.

        Args:
            branch (Conv | nn.BatchNorm2d | None): Branch to fuse.

        Returns:
            kernel (torch.Tensor): Fused kernel.
            bias (torch.Tensor): Fused bias.
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Fuse convolutions for inference by creating a single equivalent convolution."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """
    Channel-attention module for feature recalibration.

    Applies attention weights to channels based on global average pooling.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Global average pooling.
        fc (nn.Conv2d): Fully connected layer implemented as 1x1 convolution.
        act (nn.Sigmoid): Sigmoid activation for attention weights.

    References:
        https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels: int) -> None:
        """
        Initialize Channel-attention module.

        Args:
            channels (int): Number of input channels.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Channel-attended output tensor.
        """
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """
    Spatial-attention module for feature recalibration.

    Applies attention weights to spatial dimensions based on channel statistics.

    Attributes:
        cv1 (nn.Conv2d): Convolution layer for spatial attention.
        act (nn.Sigmoid): Sigmoid activation for attention weights.
    """

    def __init__(self, kernel_size=7):
        """
        Initialize Spatial-attention module.

        Args:
            kernel_size (int): Size of the convolutional kernel (3 or 7).
        """
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """
        Apply spatial attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Spatial-attended output tensor.
        """
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Combines channel and spatial attention mechanisms for comprehensive feature refinement.

    Attributes:
        channel_attention (ChannelAttention): Channel attention module.
        spatial_attention (SpatialAttention): Spatial attention module.
    """

    def __init__(self, c1, kernel_size=7):
        """
        Initialize CBAM with given parameters.

        Args:
            c1 (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel for spatial attention.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """
        Apply channel and spatial attention sequentially to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Attended output tensor.
        """
        return self.spatial_attention(self.channel_attention(x))



class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, bias=True):
        """
        Initializes the ConvLSTM cell.

        Parameters
        ----------
        in_channels: int
            Number of channels of input tensor.
        hidden_channels: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()

        self.hidden_channels = hidden_channels

        # We'll combine the input-to-gate and hidden-to-gate convolutions
        # for efficiency. This one convolution will compute all gate values.
        # It's equivalent to W_xi, W_xf, W_xc, W_xo and W_hi, W_hf, W_hc, W_ho
        self.conv = nn.Conv2d(in_channels=in_channels + hidden_channels,
                              out_channels=4 * hidden_channels,  # For i, f, o, g gates
                              kernel_size=kernel_size,
                              padding=(kernel_size // 2, kernel_size // 2),
                              bias=bias)

    def forward(self, input_tensor, cur_state):
        """
        Forward pass for a single timestep.

        Parameters
        ----------
        input_tensor: torch.Tensor
            Input tensor for the current timestep, of shape (batch, in_channels, height, width).
        cur_state: (torch.Tensor, torch.Tensor)
            A tuple containing the previous hidden state and cell state.
            h_cur shape: (batch, hidden_channels, height, width)
            c_cur shape: (batch, hidden_channels, height, width)

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            A tuple containing the new hidden state and cell state.
        """
        h_cur, c_cur = cur_state

        # Concatenate input and hidden state along the channel dimension
        combined = torch.cat([input_tensor, h_cur], dim=1)

        # Apply the single convolution
        combined_conv = self.conv(combined)

        # Split the result into the 4 gate tensors (input, forget, output, cell)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels, dim=1)

        # Calculate the gates. Note: The paper includes peephole connections
        # (W_ci, W_cf, W_co), but many modern implementations omit them for
        # simplicity. We will follow that convention here.
        i = torch.sigmoid(cc_i)  # Input gate
        f = torch.sigmoid(cc_f)  # Forget gate
        o = torch.sigmoid(cc_o)  # Output gate
        g = torch.tanh(cc_g)  # Cell gate / Candidate

        # Calculate the new cell state
        c_next = f * c_cur + i * g

        # Calculate the new hidden state
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        """
        Initializes the hidden and cell states to zeros.

        Parameters
        ----------
        batch_size: int
            The batch size.
        image_size: (int, int)
            The height and width of the input image.

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            A tuple of zero tensors for the hidden and cell states.
        """
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_channels, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_channels, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3, num_layers=1,
                 batch_first=False, bias=True):
        """
        Initializes the full ConvLSTM module.

        Parameters
        ----------
        in_channels: int
            Number of channels of input tensor.
        hidden_channels: int
            Number of channels of hidden state. This can be a list for multiple layers.
        kernel_size: (int, int)
            Size of the convolutional kernel. This can be a list for multiple layers.
        num_layers: int
            Number of ConvLSTM layers.
        batch_first: bool
            If True, then the input and output tensors are provided as (batch, seq, channel, height, width).
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTM, self).__init__()

        self.batch_first = batch_first
        self.num_layers = num_layers

        # Make sure hidden_channels and kernel_size are lists
        if not isinstance(hidden_channels, list):
            hidden_channels = [hidden_channels] * num_layers
        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * num_layers

        self.cell_list = nn.ModuleList()
        for i in range(num_layers):
            cur_in_channels = in_channels if i == 0 else hidden_channels[i - 1]
            self.cell_list.append(ConvLSTMCell(in_channels=cur_in_channels,
                                               hidden_channels=hidden_channels[i],
                                               kernel_size=kernel_size[i],
                                               bias=bias))

    def forward(self, input_tensor, hidden_state=None):
        """
        Forward pass for an entire sequence.

        Parameters
        ----------
        input_tensor: torch.Tensor
            Input tensor of shape (time, batch, channels, height, width)
            or (batch, time, channels, height, width) if batch_first=True.
        hidden_state: list of (torch.Tensor, torch.Tensor) tuples, optional
            Initial hidden and cell states for each layer. If None, states are initialized to zero.

        Returns
        -------
        layer_output_list, last_state_list:
            layer_output_list: List of lists of shape (time, batch, hidden_channels, height, width)
                               Contains the hidden states for each layer at each timestep.
            last_state_list:   List of tuples of shape (batch, hidden_channels, height, width)
                               Contains the final hidden and cell state for each layer.
        """
        if self.batch_first:
            # (b, t, c, h, w) -> (t, b, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, c, h, w = input_tensor.size()

        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(0)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[t, :, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=0)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append((h, c))

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states


class FeedbackConvLSTMBlock(nn.Module):
    """
    A wrapper for ConvLSTM that makes it compatible with YOLO's sequential model structure.
    """

    def __init__(self, c1, hidden_channels, kernel_size=3, num_layers=1):
        super().__init__()
        # This creates an instance of the ConvLSTM module
        self.conv_lstm = ConvLSTM(in_channels=c1,
                                  hidden_channels=hidden_channels,
                                  kernel_size=kernel_size,
                                  num_layers=num_layers,
                                  batch_first=True)

    def forward(self, x):
        """
        This is the forward method for your block in the YOLO model.
        """
        # 1. Prepare the input for the ConvLSTM
        x_sequence = x.unsqueeze(1)

        # 2. Call the ConvLSTM's forward method (the function you posted)
        #    and get the output.
        _, last_states = self.conv_lstm(x_sequence)

        # 3. Extract the final hidden state to pass to the next layer.
        final_hidden_state = last_states[-1][0]

        return final_hidden_state

class Concat(nn.Module):
    """
    Concatenate a list of tensors along specified dimension.

    Attributes:
        d (int): Dimension along which to concatenate tensors.
    """

    def __init__(self, dimension=1):
        """
        Initialize Concat module.

        Args:
            dimension (int): Dimension along which to concatenate tensors.
        """
        super().__init__()
        self.d = dimension

    def forward(self, x: List[torch.Tensor]):
        """
        Concatenate input tensors along specified dimension.

        Args:
            x (List[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Concatenated tensor.
        """
        return torch.cat(x, self.d)


    
class ConcatSpe(nn.Module):
    """
    Concatenate a list of tensors along specified dimension.

    Attributes:
        d (int): Dimension along which to concatenate tensors.
    """

    def __init__(self, dimension=0):
        """
        Initialize Concat module.

        Args:
            dimension (int): Dimension along which to concatenate tensors.
        """
        super().__init__()
        self.d = dimension

    def forward(self, x: List[torch.Tensor]):
        """
        Concatenate input tensors along specified dimension.

        Args:
            x (List[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Concatenated tensor.
        """

            
        cat = torch.cat(x, self.d)
        return cat


import torch
import torch.nn as nn


class TupleToTensor(nn.Module):
    """
    Safely converts a tuple of tensors (from the Detect layer) into a single tensor
    by reshaping and concatenating them.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 'x' is the tuple of tensors from the Detect layer.
        # Example: ( (b, c, 80, 80), (b, c, 40, 40), (b, c, 20, 20) )

        processed_tensors = []
        for pred in x:
            # Get the shape of the current tensor
            batch_size, channels, height, width = pred.shape

            # 1. Flatten the spatial dimensions (H, W) into a single dimension
            # Shape changes to -> (b, c, H*W)
            reshaped = pred.view(batch_size, channels, -1)

            # 2. Permute the dimensions to make it suitable for concatenation
            # Shape changes to -> (b, H*W, c)
            permuted = reshaped.permute(0, 2, 1)

            processed_tensors.append(permuted)

        # 3. Concatenate along the second dimension (the "predictions" dimension)
        return torch.cat(processed_tensors, dim=1)
class Index(nn.Module):
    """
    Returns a particular index of the input.

    Attributes:
        index (int): Index to select from input.
    """

    def __init__(self, index=0):
        """
        Initialize Index module.

        Args:
            index (int): Index to select from input.
        """
        super().__init__()
        self.index = index

    def forward(self, x: List[torch.Tensor]):
        """
        Select and return a particular index from input.

        Args:
            x (List[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Selected tensor.
        """
        return x[self.index]
