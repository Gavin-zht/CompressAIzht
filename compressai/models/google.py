# Copyright (c) 2021-2024, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#* google.py 文件 定义了几个用于图像压缩的模型类，这些模型基于不同的熵编码技术和神经网络架构。

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d
from compressai.registry import register_model

from .base import (
    SCALES_LEVELS,
    SCALES_MAX,
    SCALES_MIN,
    CompressionModel,
    get_scale_table,
)
from .utils import conv, deconv

__all__ = [
    "CompressionModel",
    "FactorizedPrior",
    "FactorizedPriorReLU",
    "ScaleHyperprior",
    "MeanScaleHyperprior",
    "JointAutoregressiveHierarchicalPriors",
    "get_scale_table",
    "SCALES_MIN",
    "SCALES_MAX",
    "SCALES_LEVELS",
]


@register_model("bmshj2018-factorized")
class FactorizedPrior(CompressionModel):
    r"""Factorized Prior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_, Int Conf. on Learning Representations
    (ICLR), 2018.
    
    功能：实现了基于完全因子化先验的图像压缩模型。
    
    属性：
    entropy_bottleneck：熵瓶颈层，用于编码和解码。
    g_a：分析变换（encoder）。
    g_s：合成变换（decoder）。
    N 和 M：网络中的通道数。
    
    方法：
    forward：前向传播，通过分析变换编码输入，通过熵瓶颈层处理，再通过合成变换解码。
    compress：压缩输入图像，返回压缩后的字符串和形状信息。
    decompress：解压缩输入字符串，返回解压缩后的图像。
    from_state_dict：从状态字典创建模型实例。

    .. code-block:: none

                  ┌───┐    y
            x ──►─┤g_a├──►─┐
                  └───┘    │
                           ▼
                         ┌─┴─┐
                         │ Q │
                         └─┬─┘
                           │
                     y_hat ▼
                           │
                           ·
                        EB :
                           ·
                           │
                     y_hat ▼
                           │
                  ┌───┐    │
        x_hat ──◄─┤g_s├────┘
                  └───┘

        EB = Entropy bottleneck

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(**kwargs)

        self.entropy_bottleneck = EntropyBottleneck(M)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )   #* 分析变换(encoder)

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )   #* 合成变换(decoder)

        self.N = N
        self.M = M

    @property
    def downsampling_factor(self) -> int:
        return 2**4

    def forward(self, x):
        y = self.g_a(x) #* 通过分析变换编码输入图像，得到特征表示 y
        y_hat, y_likelihoods = self.entropy_bottleneck(y)   #* 然后通过熵瓶颈层对 y 进行熵编码和解码，得到量化后的特征表示 y_hat 和 (特征表示的似然概率 y_likelihoods)
        x_hat = self.g_s(y_hat) #* 最后通过合成变换将 y_hat 解码为重建图像 x_hat

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
            },
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        y = self.g_a(x)
        y_strings = self.entropy_bottleneck.compress(y)#调用熵瓶颈进行熵编码的位置，将y压缩成字符串
        return {"strings": [y_strings], "shape": y.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 1
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


@register_model("bmshj2018-factorized-relu")
class FactorizedPriorReLU(FactorizedPrior):
    r"""Factorized Prior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_, Int Conf. on Learning Representations
    (ICLR), 2018.
    GDN activations are replaced by ReLU.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, 3),
        )


@register_model("bmshj2018-hyperprior")
class ScaleHyperprior(CompressionModel):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.
    
    功能：实现了基于尺度超先验的图像压缩模型。
    
    属性：
    entropy_bottleneck：熵瓶颈层，用于对超先验特征进行熵编码和解码。
    g_a 和 g_s：分析变换和合成变换，与 FactorizedPrior 类似，用于编码和解码图像特征。
    h_a 和 h_s：超先验分析变换和超先验合成变换，用于编码和解码超先验特征。
    gaussian_conditional：高斯条件层，用于对编码后的特征进行条件熵编码和解码。
    
    .. code-block:: none

                  ┌───┐    y     ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
            x ──►─┤g_a├──►─┬──►──┤h_a├──►──┤ Q ├───►───·⋯⋯·───►───┤h_s├─┐
                  └───┘    │     └───┘     └───┘        EB        └───┘ │
                           ▼                                            │
                         ┌─┴─┐                                          │
                         │ Q │                                          ▼
                         └─┬─┘                                          │
                           │                                            │
                     y_hat ▼                                            │
                           │                                            │
                           ·                                            │
                        GC : ◄─────────────────────◄────────────────────┘
                           ·                 scales_hat
                           │
                     y_hat ▼
                           │
                  ┌───┐    │
        x_hat ──◄─┤g_s├────┘
                  └───┘

        EB = Entropy bottleneck
        GC = Gaussian conditional

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(**kwargs)

        self.entropy_bottleneck = EntropyBottleneck(N)  #* self.entropy_bottleneck 表示 完全因子化的熵模型

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )       #* 分析变换模块

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )       #* 生成变换模块

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )       #* 超先验分析模块

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )       #* 超先验生成模块

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        """_summary_
        前向传播函数
        通过分析变换编码输入图像，得到特征表示 y，然后通过超先验分析变换对 y 的绝对值进行编码，得到超先验特征 z，
        接着通过熵瓶颈层对 z 进行熵编码和解码，得到量化后的超先验特征 z_hat，
        再通过超先验合成变换将 z_hat 解码为尺度参数 scales_hat，最后通过高斯条件层对 y 进行条件熵编码和解码，得到量化后的特征表示 y_hat，
        并通过合成变换将 y_hat 解码为重建图像 x_hat，返回重建图像和特征表示的似然概率。
        """
        y = self.g_a(x)     #* 通过分析变换编码输入图像，得到特征表示 y
        z = self.h_a(torch.abs(y))      #* 通过超先验分析变换对 y 的绝对值进行编码，得到超先验特征 z
        z_hat, z_likelihoods = self.entropy_bottleneck(z)   #* 通过熵瓶颈层对 z 进行熵编码和解码，得到量化后的超先验特征 z_hat 和 (量化后超先验特征表示z_hat 的似然概率 z_likelihoods)
        scales_hat = self.h_s(z_hat)        #* 超先验合成变换h_s将 z_hat 解码为尺度参数 scales_hat
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat) #* 通过高斯条件层对 y 进行条件熵编码和解码，得到量化后的特征表示 y_hat 和 (量化后特征表示y_hat 的似然概率 y_likelihoods)
        x_hat = self.g_s(y_hat)     #* 通过合成变换将 y_hat 解码为重建图像 x_hat

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }   #* 返回重建图像 x_hat 和 特征表示的似然概率(y_likelihoods 和 z_likelihoods )。

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        """_summary_
        压缩操作，相当于Encoder对输入图像进行压缩，得到码流
        
        将输入x 经过编码器进行压缩，得到码流内容(包括：特征表示y的压缩码流字符串 y_strings  和 超先验特征的压缩码流字符串 z_strings)
        Args:
            x: 输入图像

        Returns:
            压缩得到的码流(包括：特征表示y的压缩码流字符串 y_strings  和 超先验特征的压缩码流字符串 z_strings)
        """
        y = self.g_a(x)     #* 通过分析变换编码输入图像，得到特征表示 y
        z = self.h_a(torch.abs(y))      #* 通过超先验分析变换对 y 的绝对值进行编码，得到超先验特征 z

        z_strings = self.entropy_bottleneck.compress(z)     #* 熵瓶颈层对 z 进行熵编码，得到超先验特征的压缩码流字符串 z_strings
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)        #* 超先验合成变换h_s将 z_hat 解码为尺度参数 scales_hat
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)      #* 通过高斯条件层对 y 进行条件熵编码，得到特征表示y的压缩码流字符串 y_strings
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        """_summary_
        解码操作，相当于Decoder对输入的码流进行解码并计算得到重建图像
        
        Args:
            strings (_type_): _description_
            shape (_type_): _description_

        Returns:
            x_hat: 重建图像
        """
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)   #* 从码流中解码得到量化后超先验特征 z_hat
        scales_hat = self.h_s(z_hat)        #* 超先验合成变换h_s将 z_hat 解码为尺度参数 scales_hat
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


@register_model("mbt2018-mean")
class MeanScaleHyperprior(ScaleHyperprior):
    r"""
    超先验模块同时提供 均值和方差
    
    
    来源: Scale Hyperprior with non zero-mean Gaussian conditionals from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    .. code-block:: none

                  ┌───┐    y     ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
            x ──►─┤g_a├──►─┬──►──┤h_a├──►──┤ Q ├───►───·⋯⋯·───►───┤h_s├─┐
                  └───┘    │     └───┘     └───┘        EB        └───┘ │
                           ▼                                            │
                         ┌─┴─┐                                          │
                         │ Q │                                          ▼
                         └─┬─┘                                          │
                           │                                            │
                     y_hat ▼                                            │
                           │                                            │
                           ·                                            │
                        GC : ◄─────────────────────◄────────────────────┘
                           ·                 scales_hat
                           │                 means_hat
                     y_hat ▼
                           │
                  ┌───┐    │
        x_hat ──◄─┤g_s├────┘
                  └───┘

        EB = Entropy bottleneck
        GC = Gaussian conditional

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )   #*  重载定义了超先验编码器 h_a

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )   #*  重载定义了超先验解码器 h_s

    def forward(self, x):
        """_summary_
        
        输入参数: 待编码的图像x
        
        函数返回值：
        为一个字典，字典中存放了 重建图像x_hat, 以及 量化后隐特征的概率 y_likelihoods， 量化后超先验特征的概率 z_likelihoods
        """
        y = self.g_a(x) #* 分析变换得到 隐特征y
        z = self.h_a(y) #* 超先验编码器以y为输入 得到 超先验特征 z 
        z_hat, z_likelihoods = self.entropy_bottleneck(z)   #* 调用 self.entropy_bottleneck的forward方法来输出 量化后超先验特征 hat_z 以及 量化后超先验特征的概率 z_likelihoods
        gaussian_params = self.h_s(z_hat)   #* 将量化后超先验特征 hat_z 输入给超先验解码器来计算得到  量化后隐特征hat_y 所对应的高斯分布的参数(均值和方差)
        scales_hat, means_hat = gaussian_params.chunk(2, 1) #* 将均值和方差分离开
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat) #* 使用高斯条件熵模型 对 隐特征y进行编码，其中方差为scales_hat, 均值为means_hat
        #* 得到 量化后隐特征 hat_y 以及 量化后隐特征的概率 y_likelihoods
        x_hat = self.g_s(y_hat) #* 将量化后隐特征 hat_y 输入给 生成变换 计算得到重建图像 hat_x

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }   #* forward函数返回值为一个字典，字典中存放了 重建图像x_hat, 以及 量化后隐特征的概率 y_likelihoods， 量化后超先验特征的概率 z_likelihoods

    def compress(self, x):
        """_summary_

        功能： 对输入图像x进行编码压缩
        
        返回值: 
        一个字典
        - key="strings", 表示 编码得到的二进制码流，[y_strings, z_strings]
        - key="shape", 表示 超先验隐变量z的空间维度(h,w)
        
        
        """
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z) #* 使用完全因子化熵模型对 超先验隐变量z 进行压缩 得到 z对应的二进制码流 z_strings
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])    #* 使用完全因子化熵模型对 对应的二进制码流 z_strings进行解压缩得到 量化后的超先验隐变量 z_hat

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat) #* 使用 高斯条件熵模型对 隐特征y进行编码压缩，得到 y对应的二进制码流 y_strings
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


@register_model("mbt2018")
class JointAutoregressiveHierarchicalPriors(MeanScaleHyperprior):
    r"""
    
    同时使用“上下文自回归模型” 以及 “均值方差高斯条件熵模型” 的 图像编码模型
    继承自“MeanScaleHyperprior”这个 均值方差超先验网络模型
    
    Joint Autoregressive Hierarchical Priors model from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    .. code-block:: none

                  ┌───┐    y     ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
            x ──►─┤g_a├──►─┬──►──┤h_a├──►──┤ Q ├───►───·⋯⋯·───►───┤h_s├─┐
                  └───┘    │     └───┘     └───┘        EB        └───┘ │
                           ▼                                            │
                         ┌─┴─┐                                          │
                         │ Q │                                   params ▼
                         └─┬─┘                                          │
                     y_hat ▼                  ┌─────┐                   │
                           ├──────────►───────┤  CP ├────────►──────────┤
                           │                  └─────┘                   │
                           ▼                                            ▼
                           │                                            │
                           ·                  ┌─────┐                   │
                        GC : ◄────────◄───────┤  EP ├────────◄──────────┘
                           ·     scales_hat   └─────┘
                           │      means_hat
                     y_hat ▼
                           │
                  ┌───┐    │
        x_hat ──◄─┤g_s├────┘
                  └───┘

        EB = Entropy bottleneck
        GC = Gaussian conditional
        EP = Entropy parameters network
        CP = Context prediction (masked convolution)

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N=192, M=192, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
        )   #* 定义 编码器/分析变换

        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )   #* 定义 解码器/生成变换

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
        )   #* 定义 超先验编码器

        self.h_s = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )   #* 定义 超先验解码器

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )   #* 熵模型参数估计网络

        self.context_prediction = MaskedConv2d(
            M, 2 * M, kernel_size=5, padding=2, stride=1
        )   #* 上下文预测网络

        self.gaussian_conditional = GaussianConditional(None)   #* self.gaussian_conditional 表示 高斯条件熵模型
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        """_summary_

        输入参数: 
        输入图像x, 维度为(batch_size,channels, H,W)
        
        返回值:
        一个字典，
        - key为"x_hat", 表示重建图像x_hat
        - key为"likelihoods": 表示 量化后隐变量y_hat对应的概率 y_likelihoods 和 量化后超先验隐变量z_hat对应的概率 z_likelihoods
        
        """
        y = self.g_a(x) #* 输入图像x 输入 编码器得到隐变量y
        z = self.h_a(y) #* 隐变量y 输入 超先验编码器 得到 超先验隐变量 z
        z_hat, z_likelihoods = self.entropy_bottleneck(z)    #*超先验隐变量 z 输入 完全因子化的熵模型 得到 量化后超先验隐变量 z_hat 和 量化后超先验隐变量的概率 z_likelihoods
        params = self.h_s(z_hat)    #* 量化后超先验隐变量 z_hat 输入 超先验解码器self.h_s 得到 参数 params, params会用于后续的 均值和方差的预测

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )   #* 对 隐变量 y 量化得到 量化后隐变量 y_hat
        ctx_params = self.context_prediction(y_hat) #* 量化后隐变量 y_hat 输入 上下文预测模块 self.context_prediction 计算得到 上下文预测参数 ctx_params,  ctx_params会用于后续的 均值和方差的预测
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )   #* 将 params 和 ctx_params 按照特征维度进行拼接，得到 拼接后参数 con_params, 然后将其输入给 熵模型参数预测网络 self.entropy_parameters 得到高斯分布的参数  gaussian_params
        scales_hat, means_hat = gaussian_params.chunk(2, 1) #* 将 高斯分布的参数  gaussian_params 分离为 均值means_hat 和 方差 scales_hat
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)    #* 使用 均值means_hat 和 方差 scales_hat 对隐变量y进行熵编码，得到 y对应的码流 string_y 和 量化后隐变量的概率 y_likelihoods
        x_hat = self.g_s(y_hat) #* 量化后隐变量y_hat 输入给 解码器得到 重建图像 x_hat

        return {
            "x_hat": x_hat, 
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU).",
                stacklevel=2,
            )

        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
        """_summary_
        整体功能介绍
        _compress_ar 函数的作用是实现自回归（Auto-Regressive）压缩。它通过上下文预测模块和熵参数模块，对输入的特征图 y_hat 进行逐像素的压缩。
        该函数利用高斯条件模块计算每个像素的量化值，并通过 RANS 编码器（Range Asymmetric Numeral Systems）将这些量化值编码为一个字符串。
        
        参数分析：
        y_hat：填充后的特征图张量，维度为 (batch_size, M, height + 2*padding, width + 2*padding)。
        params：超先验模块生成的参数张量，维度为 (batch_size, M*2, height/32, width/32)。
        height：编码特征图的高度。
        width：编码特征图的宽度。
        kernel_size：上下文预测模块的卷积核大小。
        padding：填充大小。
        
        返回值：
        返回一个字符串，表示压缩后的特征图。
        
        整体执行逻辑:
        - 提取高斯条件模块的相关参数(cdf, cdf_length, offsets等)
        - 初始化 RANS 编码器。
        - 逐像素遍历特征图，对每个像素进行上下文预测和熵参数计算。 
        具体来说，我们遍历每一个像素位置，然后考虑该像素附近的一个小片(5x5的区域)，这个小片称为这个像素的“上下文”，对这个小片做一个二维卷积，得到一个1x1的“上下文预测值”，作为当前像素位置的上下文信息 ctx_p；
        另一方面，使用超先验解码器得到的处理后超先验参数p; 将ctx_p 和 p 拼接起来，得到拼接后特征 g= [ctx_p , p ]
        - 将拼接后特征g 输入给 self.entropy_parameters 熵参数预测网络， 得到预测的熵参数 gaussian_params，再将gaussian_params拆分为均值参数 means_hat 和 方差参数 scales_hat
        - 使用高斯条件模块对每个像素进行量化得到y_q
        - 将量化值编码为码流字符串。
        """
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()
        #* 提取高斯条件模块的累积分布函数（CDF）、CDF长度和偏移量。
        # 变量维度：
        # cdf 是一个列表，表示量化后变量的 CDF。
        # cdf_lengths 是一个列表，表示每个 CDF 的长度。
        # offsets 是一个列表，表示偏移量。
        
        
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        # 并创建用于存储符号和索引的列表。
        # 变量维度：
        # encoder 是一个 BufferedRansEncoder 对象。
        # symbols_list 和 indexes_list 是空列表，用于存储编码过程中的符号和索引。
        
        
        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        #! 注意： 尽管_compress_ar的参数y_hat在输入时不是量化版本(i.e. 非量化隐变量y)，但是后通过下面的循环将y_hat变为量化后隐变量hat{y}
        #* 并且保证，对于每一个空间坐标(h,w)，y_hat[:][:][h][w]在被用到之前，都已经被更新为量化版本
        #* 因此，本质上这里的y_hat就是"量化后隐变量hat{y}"        
        for h in range(height):
            for w in range(width):
                #* 遍历特征图y_hat的每个像素位置 (h, w), 下面我们要来预测 y_hat的(h+padding,w+padding)这个位置的像素
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]  #* 对每个像素位置，提取大小为 kernel_size x kernel_size 的局部特征图 y_crop
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )   #* 使用上下文预测模块的权重和偏置对 y_crop 进行卷积操作(F.conv2d)，得到上下文预测值 ctx_p
                #* ctx_p 表示 使用(h+padding,w+padding)的左上角那些已经解码了的像素来预测当前位置的像素的结果
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1] #* 提取超先验解码器处理后的参数 p，
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1)) #* 将参数 p和上下文预测值ctx_p拼接后，通过熵参数模块计算高斯分布的参数。
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                means_hat, scales_hat = gaussian_params.chunk(2, 1) #* 将高斯分布参数分离为均值 means_hat 和尺度 scales_hat
                #* 变量维度分析：
                # p 的维度为 (batch_size, M*2, 1, 1)。
                # gaussian_params 的维度为 (batch_size, M*2, 1, 1)。
                # means_hat 和 scales_hat 的维度均为 (batch_size, M)。
                indexes = self.gaussian_conditional.build_indexes(scales_hat)    #* 使用高斯条件模块构建索引 indexes。
                y_crop = y_crop[:, :, padding, padding] # 提取中心像素值 y_crop。
                y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)  #* 使用高斯条件模块对中心像素值进行量化，得到量化值 y_q。
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat #* 更新特征图 y_hat 的(h+padding, w+padding)位置，
                #! i.e. y_hat的(h+padding, w+padding)位置已经被更新量化后版本了，它还尚未被用到，在用到它的时候他已经是代表 量化后隐变量hat{y}

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )   #* 使用 RANS 编码器将符号和索引编码为字符串。

        string = encoder.flush()    #* string 是一个字节序列，表示压缩后的特征图。
        return string

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU).",
                stacklevel=2,
            )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )
        #* 初始化 y_hat 为零张量，并添加适当的填充，以便可以直接处理子张量。
        #* y_hat 的形状为 (batch_size, channels, y_height + 2 * padding, y_width + 2 * padding)

        for i, y_string in enumerate(strings[0]):   #* 遍历 strings[0] 中的每个编码字符串 y_string。
            self._decompress_ar(
                y_string,
                y_hat[i : i + 1],   #* y_hat[i] 表示y_hat当前小批量中的第i个样本，y_hat形状为(C,H,W)
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )   #* 调用 _decompress_ar 方法，逐元素解码 y_string，并将结果存储在 y_hat 中。

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))  #* 使用 F.pad 方法去除 y_hat 的填充部分，恢复原始形状。
        #* y_hat 的最终形状为 (batch_size, channels, y_height, y_width)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def _decompress_ar(
        self, y_string, y_hat, params, height, width, kernel_size, padding
    ):
        """_summary_
        1. 功能概述
        _decompress_ar 是一个自回归解码函数，用于逐像素解码输入的字节流 y_string，并恢复原始张量 y_hat。
        它利用上下文预测和高斯条件模型来解码每个像素值。这种方法通常用于图像压缩和视频编码中的熵解码。
        
        2. 参数分析
        y_string：编码后的字节流，用于解码。
        y_hat：初始化为零的张量，用于存储解码后的结果。
        params：超参数张量，包含解码所需的额外信息。
        height：解码后张量的高度。
        width：解码后张量的宽度。
        kernel_size：上下文预测卷积核的大小。
        padding：填充大小，用于处理边界情况。
        
        3. 返回值
        该方法没有返回值，但会直接修改输入张量 y_hat，将解码后的值填充到其中。
        
        4. 整体执行逻辑
        _decompress_ar 的主要逻辑包括：
        - 初始化解码器：创建 RansDecoder 实例并设置输入字节流。
        - 逐像素解码：通过上下文预测和高斯条件模型逐像素解码输入字节流。
        - 更新解码张量：将解码后的值填充到 y_hat 中。
        """
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        decoder = RansDecoder() #* 创建一个 RansDecoder 实例，用于解码输入字节流。
        decoder.set_stream(y_string)    #* 调用 set_stream 方法，将输入字节流 y_string 设置为解码器的输入。

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        #* 下面这个循环，用(h,w)位置一个5x5的小片作为“上下文”来预测 (h+2,w+2)位置的y_hat
        for h in range(height):
            for w in range(width):
                #* 遍历每个像素位置 (h, w)
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]  #* 从 y_hat 中裁剪出以 (h, w) 为中心的 kernel_size x kernel_size 的子张量 y_crop。
                ctx_p = F.conv2d(
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )   #* 使用上下文预测卷积层 self.context_prediction 对 y_crop 进行卷积，得到上下文预测值 ctx_p。
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]  #* 从 params 中提取以 (h, w) 为中心的 1x1 子张量 p
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1)) #* 将 p 和 ctx_p 拼接起来，通过熵参数预测网络 self.entropy_parameters 得到高斯参数 means_hat 和 scales_hat。
                means_hat, scales_hat = gaussian_params.chunk(2, 1)
                #* means_hat 表示 高斯分布的均值， scales_hat 表示高斯分布的方差
                indexes = self.gaussian_conditional.build_indexes(scales_hat)   #* 使用 self.gaussian_conditional.build_indexes 构建索引 indexes
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )   #* 调用 decoder(RansDecoder).decode_stream 解码当前像素值 rv(i.e. 量化后隐特征y_hat)
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv, means_hat)    #* 对rv进行反量化操作，rv = rv + means_hat

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv  #* 将解码后的值 rv 填充到 y_hat 的相应位置, i.e. 空间位置为 (h+padding, w+padding)
