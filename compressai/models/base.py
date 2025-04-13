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


#* base.py 定义了两个与图像压缩相关的基础类：CompressionModel(图像压缩模型的基础类) 和 SimpleVAECompressionModel(基于简单的变分自编码器（VAE）压缩模型, 继承自CompressionModel)
#* 之后实现的图像压缩模型都是继承自这个CompressionModel 类

import math
import warnings

from typing import cast

import torch
import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.latent_codecs import LatentCodec
from compressai.models.utils import remap_old_keys, update_registered_buffers

__all__ = [
    "CompressionModel",
    "SimpleVAECompressionModel",
    "get_scale_table",
    "SCALES_MIN",
    "SCALES_MAX",
    "SCALES_LEVELS",
]


# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    """Returns table of logarithmically scales."""
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class CompressionModel(nn.Module):
    """Base class for constructing an auto-encoder with any number of
    EntropyBottleneck or GaussianConditional modules.
    
    功能：作为构建自编码器的基类，可以包含任意数量的 EntropyBottleneck 或 GaussianConditional 模块。
    
    属性：
    entropy_bottleneck：熵瓶颈层，用于对编码后的特征进行熵编码和解码。该属性是可选的，如果在初始化时传入 entropy_bottleneck_channels 参数，则会创建一个熵瓶颈层。    
    
    
    
    """

    def __init__(self, entropy_bottleneck_channels=None, init_weights=None):
        super().__init__()

        if entropy_bottleneck_channels is not None:     #* 如果传入 entropy_bottleneck_channels 参数，则创建一个熵瓶颈层，并发出弃用警告。
            warnings.warn(
                "The entropy_bottleneck_channels parameter is deprecated. "
                "Create an entropy_bottleneck in your model directly instead:\n\n"
                "class YourModel(CompressionModel):\n"
                "    def __init__(self):\n"
                "        super().__init__()\n"
                "        self.entropy_bottleneck = "
                "EntropyBottleneck(entropy_bottleneck_channels)\n",
                DeprecationWarning,
                stacklevel=2,
            )   
            self.entropy_bottleneck = EntropyBottleneck(entropy_bottleneck_channels)

        if init_weights is not None:
            warnings.warn(
                "The init_weights parameter was removed as it was never functional.",
                DeprecationWarning,
                stacklevel=2,
            )

    def load_state_dict(self, state_dict, strict=True):
        for name, module in self.named_modules():
            if not any(x.startswith(name) for x in state_dict.keys()):
                continue

            if isinstance(module, EntropyBottleneck):
                update_registered_buffers(
                    module,
                    name,
                    ["_quantized_cdf", "_offset", "_cdf_length"],
                    state_dict,
                )
                state_dict = remap_old_keys(name, state_dict)

            if isinstance(module, GaussianConditional):
                update_registered_buffers(
                    module,
                    name,
                    ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
                    state_dict,
                )

        return nn.Module.load_state_dict(self, state_dict, strict=strict)

    def update(self, scale_table=None, force=False, update_quantiles: bool = False):
        """Updates EntropyBottleneck and GaussianConditional CDFs.

        #! 注意: 当模型训练完，在评估之前，需要执行一下这个update函数来更新熵模型中的CDF函数
        原因: 因为在模型评估的时候，会使用 rv = self.entropy_coder.encode_with_indexes 来进行真正的压缩；
        在encode_with_indexes函数中，会使用self._quantized_cdf 来作为量化后隐变量的概率分布函数
        实际上，我们在训练的时候，都是直接使用 非量化隐变量y的累计分布函数CDF 去计算 量化隐变量hat_y 的概率分布CDF； 然后得到量化后隐变量的概率值;
        在这个过程中，我们并不需要使用self._quantized_cdf
        但是我们在模型评估/测试的时候，却要使用self._quantized_cdf来进行真实的压缩，因此我们需要在测试/评估之前，将self._quantized_cdf更新到最新的正确值。
            
        功能：更新模型中的 EntropyBottleneck 和 GaussianConditional 模块的累积分布函数（CDF）:self._quantized_cdf
        
        
        
        
        参数：
        scale_table：缩放表，用于初始化高斯分布，默认为 None。
        force：是否强制更新，默认为 False。
        update_quantiles：是否快速更新分位数，默认为 False。
        
        实现：
        如果 scale_table 为 None，则调用 get_scale_table 函数生成默认的缩放表。
        遍历模型的所有模块，如果模块是 EntropyBottleneck 或 GaussianConditional，则调用其 update 方法更新 CDF。
        返回是否至少有一个模块被更新。




        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            scale_table (torch.Tensor): table of scales (i.e. stdev)
                for initializing the Gaussian distributions
                (default: 64 logarithmically spaced scales from 0.11 to 256)
            force (bool): overwrite previous values (default: False)
            update_quantiles (bool): fast update quantiles (default: False)

        Returns:
            updated (bool): True if at least one of the modules was updated.
        """
        if scale_table is None:
            scale_table = get_scale_table()
        updated = False
        for _, module in self.named_modules():
            #* 遍历模型CompressionModel的所有模块 module，如果模块module是 EntropyBottleneck 或 GaussianConditional，则调用其 update 方法更新 CDF。
            if isinstance(module, EntropyBottleneck):
                updated |= module.update(force=force, update_quantiles=update_quantiles)
            if isinstance(module, GaussianConditional):
                updated |= module.update_scale_table(scale_table, force=force)
        
        
        return updated  #* 返回updated, updated 记录 在当前模型CompressionModel中是否有一个模块被更新

    def aux_loss(self) -> Tensor:
        r"""Returns the total auxiliary loss over all ``EntropyBottleneck``\s.


        功能：计算所有 EntropyBottleneck 模块的辅助损失。
        
        返回值：辅助损失，类型为 Tensor。
        
        实现：
        遍历模型的所有模块，累加所有 EntropyBottleneck 模块的损失。


        与“net”优化器使用的主要“net”损失不同，“aux”损失仅由“aux”优化器使用，用于更新仅 `EntropyBottleneck.quantiles` 参数。事实上，“aux”损失完全不依赖于图像数据。

        “aux”损失的目的是确定给定分布的大部分质量所在的范围及其中位数（即50%概率）。也就是说，对于给定的分布，“aux”损失会朝着满足以下条件的方向收敛（对于某个选定的 `tail_mass` 概率）：

        - `cdf(quantiles[0]) = tail_mass / 2`
        - `cdf(quantiles[1]) = 0.5`
        - `cdf(quantiles[2]) = 1 - tail_mass / 2`

        这确保了具体的 `_quantized_cdf` 主要在有限支持的区域内操作。任何超出此范围的符号必须使用不涉及 `_quantized_cdf` 的替代方法进行编码。
        幸运的是，可以选择一个足够小的 `tail_mass` 概率，使得这种情况很少发生。
        重要的是，我们使用的 `_quantized_cdf` 具有较小的有限支持；否则，熵编码的运行性能会受到影响。因此，`tail_mass` 也不应设置得过小！



        In contrast to the primary "net" loss used by the "net"
        optimizer, the "aux" loss is only used by the "aux" optimizer to
        update *only* the ``EntropyBottleneck.quantiles`` parameters. In
        fact, the "aux" loss does not depend on image data at all.

        The purpose of the "aux" loss is to determine the range within
        which most of the mass of a given distribution is contained, as
        well as its median (i.e. 50% probability). That is, for a given
        distribution, the "aux" loss converges towards satisfying the
        following conditions for some chosen ``tail_mass`` probability:

        * ``cdf(quantiles[0]) = tail_mass / 2``
        * ``cdf(quantiles[1]) = 0.5``
        * ``cdf(quantiles[2]) = 1 - tail_mass / 2``

        This ensures that the concrete ``_quantized_cdf``\s operate
        primarily within a finitely supported region. Any symbols
        outside this range must be coded using some alternative method
        that does *not* involve the ``_quantized_cdf``\s. Luckily, one
        may choose a ``tail_mass`` probability that is sufficiently
        small so that this rarely occurs. It is important that we work
        with ``_quantized_cdf``\s that have a small finite support;
        otherwise, entropy coding runtime performance would suffer.
        Thus, ``tail_mass`` should not be too small, either!
        """
        loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return cast(Tensor, loss)


class SimpleVAECompressionModel(CompressionModel):
    """Simple VAE model with arbitrary latent codec.

    功能：基于简单的变分自编码器（VAE）压缩模型，包含任意的潜在编码器，继承自CompressionModel基类.
    
    属性：
    g_a：分析变换（encoder），用于将输入图像编码为潜在表示。
    g_s：合成变换（decoder），用于将潜在表示解码为重建图像。
    latent_codec：潜在编码器，用于对潜在表示进行编码和解码。




    .. code-block:: none

               ┌───┐  y  ┌────┐ y_hat ┌───┐
        x ──►──┤g_a├──►──┤ lc ├───►───┤g_s├──►── x_hat
               └───┘     └────┘       └───┘
    """

    g_a: nn.Module  #* 定义了SimpleVAECompressionModel类的一个属性g_a(是一个nn.Module)， 作为分析变换g_a
    g_s: nn.Module  #* 定义了SimpleVAECompressionModel类的一个属性g_s(是一个nn.Module)， 作为生成变换g_s
    latent_codec: LatentCodec  #* 定义了SimpleVAECompressionModel类的一个属性latent_codec(是一个LatentCodec)， 作为潜在编码器latent_codec

    def __getitem__(self, key: str) -> LatentCodec:
        return self.latent_codec[key]

    def forward(self, x):
        """_summary_
        功能：前向传播。
        
        参数：x，输入图像。
        
        返回值：包含重建图像和潜在表示似然概率的字典。

        实现：
        通过分析变换编码输入图像，得到潜在表示 y。
        通过潜在编码器处理潜在表示 y，得到量化后的潜在表示 y_hat 和似然概率。
        通过合成变换将 y_hat 解码为重建图像 x_hat。
        返回包含 x_hat 和似然概率的字典。
        
        """
        y = self.g_a(x)
        y_out = self.latent_codec(y)
        y_hat = y_out["y_hat"]
        x_hat = self.g_s(y_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": y_out["likelihoods"],
        }

    def compress(self, x):
        """_summary_
        功能：压缩输入图像。
        
        参数：x，输入图像。
        
        返回值：压缩后的输出。
        
        实现：
        通过分析变换编码输入图像，得到潜在表示 y。
        通过潜在编码器压缩潜在表示 y，得到压缩后的输出。
        """
        y = self.g_a(x)
        outputs = self.latent_codec.compress(y)
        return outputs

    def decompress(self, *args, **kwargs):
        """_summary_
        功能：解压缩输入。
        
        参数：可变参数和关键字参数，具体取决于潜在编码器的解压缩方法。
        
        返回值：包含重建图像的字典。
        
        实现：
        通过潜在编码器解压缩输入，得到量化后的潜在表示 y_hat。
        通过合成变换将 y_hat 解码为重建图像 x_hat，并进行裁剪。
        返回包含 x_hat 的字典。
        """
        y_out = self.latent_codec.decompress(*args, **kwargs)
        y_hat = y_out["y_hat"]
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {
            "x_hat": x_hat,
        }
