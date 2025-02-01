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


#* entropy_models.py 定义了几个与熵编码相关的工具类和函数，主要用于图像压缩中的熵模型

import warnings

from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from compressai._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf
from compressai.ops import LowerBound


class _EntropyCoder:
    """Proxy class to an actual entropy coder class.
    
    功能：代理类，用于封装实际的熵编码器类。
    
    属性：
    name：熵编码器的名称。
    _encoder：编码器实例。
    _decoder：解码器实例。
    
    方法：
    encode_with_indexes 和 decode_with_indexes：分别用于编码和解码数据，使用索引信息。    
    
    
    """

    def __init__(self, method):
        if not isinstance(method, str):
            raise ValueError(f'Invalid method type "{type(method)}"')

        from compressai import available_entropy_coders

        if method not in available_entropy_coders():
            methods = ", ".join(available_entropy_coders())
            raise ValueError(
                f'Unknown entropy coder "{method}"' f" (available: {methods})"
            )

        if method == "ans":
            from compressai import ans
            #当method是ans，从compressai模块导入ANS编解码器
            encoder = ans.RansEncoder()
            decoder = ans.RansDecoder()
        elif method == "rangecoder":
            import range_coder

            encoder = range_coder.RangeEncoder()
            decoder = range_coder.RangeDecoder()

        self.name = method
        self._encoder = encoder
        self._decoder = decoder

    def encode_with_indexes(self, *args, **kwargs):     #* 调用编码器实例的encode_with_indexes(*args, **kwargs)进行编码操作
        """
        依次将以下信息传入(*args)中
        一维list做symbols, 
        一个一维list做indexes表示每个symbol对应的CDF索引, 
        一个二维list做cdf（其中每一个维度都应该递增分布）,
        一个一维list对应于cdf中每个1维数据的长度 cdf_lengths,
        一个一维list对应于cdf中每个1维数据的offsets。

        self包含init时定义的编解码器

        """
        #s=self._encoder.encode_with_indexes(*args, **kwargs)
        #print("test")
        #return s
        return self._encoder.encode_with_indexes(*args, **kwargs)

    def decode_with_indexes(self, *args, **kwargs):     #* 调用编码器实例的decode_with_indexes(*args, **kwargs)进行解码操作
        return self._decoder.decode_with_indexes(*args, **kwargs)


def default_entropy_coder():
    """_summary_
    功能：返回默认的熵编码器。
    
    实现：从 compressai 模块调用 get_entropy_coder 函数，获取并返回默认的熵编码器。
    """
    from compressai import get_entropy_coder

    return get_entropy_coder()


def pmf_to_quantized_cdf(pmf: Tensor, precision: int = 16) -> Tensor:
    """_summary_
    功能：将概率密度函数（PMF）转换为累积概率密度函数（CDF）。
    
    参数：
    pmf：概率密度函数，类型为 Tensor。
    precision：精度，类型为 int，默认值为16。
    
    实现：
    调用 _pmf_to_quantized_cdf 函数，传入 pmf.tolist() 和 precision，得到量化CDF的列表表示。
    将列表表示转换为 torch.IntTensor 类型，返回量化CDF。
    """
    cdf = _pmf_to_quantized_cdf(pmf.tolist(), precision)
    cdf = torch.IntTensor(cdf)
    return cdf


def _forward(self, *args: Any) -> Any:
    raise NotImplementedError()


class EntropyModel(nn.Module):
    r"""Entropy model base class.

    功能：熵模型基类，用于图像压缩中的熵编码。
    
    属性：
    entropy_coder：熵编码器实例。
    entropy_coder_precision：熵编码器精度。
    use_likelihood_bound：是否使用似然下界。
    _offset、_quantized_cdf、_cdf_length：用于存储偏移、量化 CDF 和 CDF 长度。


    Args:
        likelihood_bound (float): minimum likelihood bound
        entropy_coder (str, optional): set the entropy coder to use, use default
            one if None
        entropy_coder_precision (int): set the entropy coder precision
    """

    def __init__(
        self,
        likelihood_bound: float = 1e-9,
        entropy_coder: Optional[str] = None,
        entropy_coder_precision: int = 16,
    ):
        """_summary_
        1. 功能概述
        __init__ 方法是 EntropyModel 类的初始化方法，用于设置熵模型的基本属性和参数。
        它初始化熵编码器、似然下界，并注册一些缓冲区，这些缓冲区将在后续的 update 方法中被填充。
        
        2. 参数分析
        likelihood_bound: float：似然下界，用于防止似然值过小。默认值为 1e-9。
        entropy_coder: Optional[str]：熵编码器的名称。如果为 None，则使用默认的熵编码器。默认值为 None。
        entropy_coder_precision: int：熵编码器的精度，通常是一个整数。默认值为 16。
        
        3. 返回值分析
        __init__ 方法没有返回值，但它初始化了以下属性：
        self.entropy_coder：熵编码器实例。
        self.entropy_coder_precision：熵编码器的精度。
        self.use_likelihood_bound：是否使用似然下界。
        self.likelihood_lower_bound：似然下界实例（如果 likelihood_bound > 0）。
        self._offset：偏移量缓冲区。
        self._quantized_cdf：量化 CDF 缓冲区。
        self._cdf_length：CDF 长度缓冲区。
        
        4. 整体执行逻辑
        __init__ 方法的主要逻辑包括：
        熵编码器初始化：如果未指定熵编码器，则使用默认的熵编码器。
        熵编码器精度设置：将传入的精度值转换为整数并存储。
        似然下界设置：如果 likelihood_bound 大于 0，则初始化似然下界。
        缓冲区注册：注册 _offset、_quantized_cdf 和 _cdf_length 缓冲区，这些缓冲区将在后续的 update 方法中被填充。
        """
        super().__init__()

        if entropy_coder is None:
            entropy_coder = default_entropy_coder() #* 熵编码器使用默认的ANS(非对称数字系统)编码器
        self.entropy_coder = _EntropyCoder(entropy_coder)
        self.entropy_coder_precision = int(entropy_coder_precision) #* 设定熵编码器的精度

        self.use_likelihood_bound = likelihood_bound > 0
        # 检查 likelihood_bound 是否大于 0。如果大于 0，则设置 self.use_likelihood_bound 为 True，并初始化 LowerBound 实例。
        # 如果 likelihood_bound 小于或等于 0，则不使用似然下界。        
        if self.use_likelihood_bound:
            self.likelihood_lower_bound = LowerBound(likelihood_bound)  #* 设定self.likelihood_lower_bound 表示 似然的下界

        # to be filled on update()
        self.register_buffer("_offset", torch.IntTensor())
        self.register_buffer("_quantized_cdf", torch.IntTensor())   #* 量化后CDF(累积分布函数)
        self.register_buffer("_cdf_length", torch.IntTensor())      #* cdf的长度/单位数
        # 使用 register_buffer 方法注册 _offset、_quantized_cdf 和 _cdf_length 缓冲区。这些缓冲区将在后续的 update 方法中被填充。
        # 这些缓冲区的初始值为空的 torch.IntTensor。

        #*self.register_buffer 是 PyTorch 中的一个方法，用于在模块中注册一个缓冲区（buffer）。缓冲区通常用于存储不参与梯度计算的张量，例如模型的统计信息、固定参数或其他需要在训练过程中保持不变的数据。     
        # register_buffer(name: str, tensor: Optional[Tensor], persistent: bool = True)
        # name: str：缓冲区的名称，必须是一个字符串。
        # tensor: Optional[Tensor]：要注册的张量。如果为 None，则不会注册任何缓冲区。
        # persistent: bool = True：是否将缓冲区保存到模块的状态字典中。默认值为 True，表示缓冲区会保存到状态字典中，可以在模型保存和加载时使用。   
        

    def __getstate__(self):
        attributes = self.__dict__.copy()
        attributes["entropy_coder"] = self.entropy_coder.name
        return attributes

    def __setstate__(self, state):
        self.__dict__ = state
        self.entropy_coder = _EntropyCoder(self.__dict__.pop("entropy_coder"))

    @property
    def offset(self):
        """_summary_
        返回熵模型的self.offset
        """
        return self._offset

    @property
    def quantized_cdf(self):
        """_summary_
        返回熵模型的self._quantized_cdf
        """        
        return self._quantized_cdf

    @property
    def cdf_length(self):
        return self._cdf_length

    # See: https://github.com/python/mypy/issues/8795
    forward: Callable[..., Any] = _forward

    def quantize(
        self, inputs: Tensor, mode: str, means: Optional[Tensor] = None
    ) -> Tensor:
        """_summary_
    功能：对输入张量进行量化。
    
    参数：
    inputs：输入张量，类型为 Tensor。
    mode：量化模式，可选值为 "noise"、"dequantize" 或 "symbols"。
    means：可选参数，均值张量，类型为 Optional[Tensor]。
    
    实现：
    检查 mode 是否有效，如果无效则抛出 ValueError。
    根据 mode 的值，分别进行以下操作：
    - "noise"：在输入张量上添加均匀分布的噪声，然后返回结果。
    - "dequantize"：此时直接返回torch.round(inputs - means) + means, 也就约等于对inputs四舍五入
    - "symbols"：将输入张量减去 means（如果 means 不为 None），然后进行四舍五入，转换为整数类型，返回结果。i.e. 此时直接返回torch.round(inputs - means)
        """
        if mode not in ("noise", "dequantize", "symbols"):
            raise ValueError(f'Invalid quantization mode: "{mode}"')

        if mode == "noise":
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)  #* 定义一个噪声张量 noise, 其大小与inputs相同，noise的每一个元素都是从均匀分布[-0.5, 0.5]上随机采样得到的。
            inputs = inputs + noise #* 在inputs上加上噪声，作为量化值的近似替代。
            return inputs
        #* 至此，表明mode参数不为"noise"
        outputs = inputs.clone()
        if means is not None:
            outputs -= means    #* 如果传入了均值参数，就先减去均值

        outputs = torch.round(outputs)  #* torch.round 是 PyTorch 中的一个函数，用于对输入张量的每个元素进行四舍五入操作。它将每个元素四舍五入到最接近的整数。

        if mode == "dequantize":
            if means is not None:
                outputs += means
            return outputs  

        assert mode == "symbols", mode
        outputs = outputs.int()
        return outputs

    def _quantize(
        self, inputs: Tensor, mode: str, means: Optional[Tensor] = None
    ) -> Tensor:
        """_summary_
    功能：已弃用的量化函数，建议使用 quantize 函数代替。
    
    参数：与 quantize 函数相同。
    
    实现：调用 quantize 函数，并发出弃用警告。
        """
        warnings.warn("_quantize is deprecated. Use quantize instead.", stacklevel=2)
        return self.quantize(inputs, mode, means)

    @staticmethod
    def dequantize(
        inputs: Tensor, means: Optional[Tensor] = None, dtype: torch.dtype = torch.float
    ) -> Tensor:
        """_summary_
        功能：对输入张量进行反量化。
        
        参数：
        inputs：输入张量，类型为 Tensor。
        means：可选参数，均值张量，类型为 Optional[Tensor]。
        dtype：反量化输出的类型，类型为 torch.dtype，默认为 torch.float。
        
        实现：
        如果 means 不为 None，则将输入张量转换为与 means 相同的类型，并与 means 相加；
        否则将输入张量转换为指定的 dtype，返回结果。
        
        返回值：
        inputs的反量化值(inputs + means)
        
        """
        if means is not None:
            outputs = inputs.type_as(means)
            outputs += means
        else:
            outputs = inputs.type(dtype)
        return outputs

    @classmethod
    def _dequantize(cls, inputs: Tensor, means: Optional[Tensor] = None) -> Tensor:
        """_summary_
        功能：已弃用的反量化函数，建议使用 dequantize 函数代替。
        
        参数：与 dequantize 函数相同。
        
        实现：调用 dequantize 函数，并发出弃用警告。

        """
        warnings.warn("_dequantize. Use dequantize instead.", stacklevel=2)
        return cls.dequantize(inputs, means)

    def _pmf_to_cdf(self, pmf, tail_mass, pmf_length, max_length):
        """_summary_

        功能：将概率密度函数（PMF）转换为累积分布函数（CDF）。
        
        参数：
        pmf：概率质量函数。
        tail_mass：尾部质量。
        pmf_length：PMF的长度。
        max_length：最大长度。
        
        实现：
        创建一个大小为 (len(pmf_length), max_length + 2) 的零张量 cdf。
        遍历 pmf 中的每个概率分布，将其与尾部质量拼接，然后调用 pmf_to_quantized_cdf 函数进行转换，将结果存储在 cdf 中。
        返回 cdf。


        Args:
            pmf (_type_): _description_
            tail_mass (_type_): _description_
            pmf_length (_type_): _description_
            max_length (_type_): _description_

        Returns:
            _type_: _description_
        """
        cdf = torch.zeros(
            (len(pmf_length), max_length + 2), dtype=torch.int32, device=pmf.device
        )
        for i, p in enumerate(pmf):
            prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
            _cdf = pmf_to_quantized_cdf(prob, self.entropy_coder_precision)
            cdf[i, : _cdf.size(0)] = _cdf
        return cdf

    def _check_cdf_size(self):
        if self._quantized_cdf.numel() == 0:
            raise ValueError("Uninitialized CDFs. Run update() first")

        if len(self._quantized_cdf.size()) != 2:
            raise ValueError(f"Invalid CDF size {self._quantized_cdf.size()}")

    def _check_offsets_size(self):
        if self._offset.numel() == 0:
            raise ValueError("Uninitialized offsets. Run update() first")

        if len(self._offset.size()) != 1:
            raise ValueError(f"Invalid offsets size {self._offset.size()}")

    def _check_cdf_length(self):
        if self._cdf_length.numel() == 0:
            raise ValueError("Uninitialized CDF lengths. Run update() first")

        if len(self._cdf_length.size()) != 1:
            raise ValueError(f"Invalid offsets size {self._cdf_length.size()}")

    def compress(self, inputs, indexes, means=None):
        """
        Compress input tensors to char strings.

        功能：将输入张量压缩为码流字符串。
        
        参数：
        inputs：输入张量，类型为 torch.Tensor, 维度为(batch_size , channels, height, width)
        indexes：CDF索引张量，类型为 torch.IntTensor, 维度为(batch_size , channels, height, width)
        means：可选参数，均值张量，类型为 Optional[torch.Tensor]，维度为(batch_size , channels, height, width)

        返回值：压缩后的字符串(码流)列表。
        
        实现过程
        使用 quantize 方法将输入张量 inputs 量化为符号张量 symbols。
        检查 inputs 的维度是否至少为2，如果不是则抛出 ValueError。
        检查 inputs 和 indexes 的大小是否相同，如果不同则抛出 ValueError。
        调用 _check_cdf_size、_check_cdf_length 和 _check_offsets_size 方法，检查熵模型的 CDF、偏移量和 CDF 长度是否已初始化。
        遍历 symbols 的每个批次，使用 entropy_coder 的 encode_with_indexes 方法对符号进行编码，将编码结果存储在 strings 列表中。
        返回 strings 列表，其中每个元素是一个压缩后的字符串。        

        """
        symbols = self.quantize(inputs, "symbols", means)   #* 使用 quantize 方法将输入张量 inputs 量化为符号张量 symbols。量化模式为 "symbols"，如果 means 不为 None，则在量化前减去 means。

        if len(inputs.size()) < 2:  #* 检查 inputs 的维度是否至少为2，如果不是则抛出 ValueError。
            raise ValueError(
                "Invalid `inputs` size. Expected a tensor with at least 2 dimensions."
            )

        if inputs.size() != indexes.size():     #* 检查 inputs 和 indexes 的大小是否相同，如果不同则抛出 ValueError
            raise ValueError("`inputs` and `indexes` should have the same size.")

        #* 调用 _check_cdf_size、_check_cdf_length 和 _check_offsets_size 方法，检查熵模型的 CDF、偏移量和 CDF 长度是否已初始化。
        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()

        strings = []
        for i in range(symbols.size(0)):
            #* 遍历 symbols 的每个批次，使用 entropy_coder 的 encode_with_indexes 方法对符号进行编码。编码时传入符号、索引、量化 CDF、CDF 长度和偏移量。

            rv = self.entropy_coder.encode_with_indexes(
                symbols[i].reshape(-1).int().tolist(),
                indexes[i].reshape(-1).int().tolist(),
                self._quantized_cdf.tolist(),
                self._cdf_length.reshape(-1).int().tolist(),
                self._offset.reshape(-1).int().tolist(),
            )
            #传入参数时将torch.Tensor格式转换为list。
            strings.append(rv)  #* 将编码结果存储在 strings 列表中。
        return strings  #* 返回 strings 列表，其中每个元素是一个压缩后的字符串。

    def decompress(
        self,
        strings: str,
        indexes: torch.IntTensor,
        dtype: torch.dtype = torch.float,
        means: torch.Tensor = None,
    ):
        """
        Decompress char strings to tensors.

        参数:
            strings (str): 压缩后的字符串列表。
            indexes (torch.IntTensor): CDF索引张量。
            dtype (torch.dtype): 反量化输出的类型，默认为 torch.float。
            means (torch.Tensor, optional): 可选参数，均值张量。


        Args:
            strings (str): compressed tensors
            indexes (torch.IntTensor): tensors CDF indexes
            dtype (torch.dtype): type of dequantized output
            means (torch.Tensor, optional): optional tensor means
        """

        if not isinstance(strings, (tuple, list)):
            raise ValueError("Invalid `strings` parameter type.")

        if not len(strings) == indexes.size(0):
            raise ValueError("Invalid strings or indexes parameters")

        if len(indexes.size()) < 2:
            raise ValueError(
                "Invalid `indexes` size. Expected a tensor with at least 2 dimensions."
            )

        #* 调用 _check_cdf_size、_check_cdf_length 和 _check_offsets_size 方法，检查熵模型的 CDF、偏移量和 CDF 长度是否已初始化。
        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()

        if means is not None:
            if means.size()[:2] != indexes.size()[:2]:
                raise ValueError("Invalid means or indexes parameters")
            if means.size() != indexes.size():
                for i in range(2, len(indexes.size())):
                    if means.size(i) != 1:
                        raise ValueError("Invalid means parameters")

        cdf = self._quantized_cdf
        outputs = cdf.new_empty(indexes.size()) #* 创建一个与 indexes 大小相同的空张量 outputs，用于存储解码后的值。

        for i, s in enumerate(strings):
            #* 遍历 strings 的每个字符串 s：
            #* 使用 entropy_coder 的 decode_with_indexes 方法对字符串 s 进行解码。解码时传入字符串、索引、量化 CDF、CDF 长度和偏移量。
            #* 将解码结果存储在 outputs 张量中。
            values = self.entropy_coder.decode_with_indexes(
                s,
                indexes[i].reshape(-1).int().tolist(),
                cdf.tolist(),
                self._cdf_length.reshape(-1).int().tolist(),
                self._offset.reshape(-1).int().tolist(),
            )
            outputs[i] = torch.tensor(
                values, device=outputs.device, dtype=outputs.dtype
            ).reshape(outputs[i].size())
        outputs = self.dequantize(outputs, means, dtype)    #* 使用 dequantize 方法对 outputs 进行反量化。反量化时传入 means 和 dtype。
        return outputs


class EntropyBottleneck(EntropyModel):
    r"""
    EntropyBottleneck(熵瓶颈层) 类是 EntropyModel 的一个子类，实现了熵瓶颈层，用于图像压缩中的熵编码。
    EntropyBottleneck 就是“基于完全因子化的熵模型”
    这个类基于 J. Ballé 等人在论文 "Variational image compression with a scale hyperprior" 中提出的方法。    
    
    
    Entropy bottleneck layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.

    This is a re-implementation of the entropy bottleneck layer in
    *tensorflow/compression*. See the original paper and the `tensorflow
    documentation
    <https://github.com/tensorflow/compression/blob/v1.3/docs/entropy_bottleneck.md>`__
    for an introduction.
    """

    _offset: Tensor

    def __init__(
        self,
        channels: int,
        *args: Any,
        tail_mass: float = 1e-9,
        init_scale: float = 10,
        filters: Tuple[int, ...] = (3, 3, 3, 3),
        **kwargs: Any,
    ):
        """_summary_
        参数：
        channels：通道数。
        tail_mass：尾部质量，默认为 1e-9。
        init_scale：初始化缩放因子，默认为 10。
        filters：滤波器大小，默认为 (3, 3, 3, 3)。
        
        实现：
        调用父类 EntropyModel 的初始化方法。
        初始化通道数、滤波器大小、初始化缩放因子和尾部质量。
        创建参数列表 matrices、biases 和 factors，用于存储网络的权重、偏置和因子。
        初始化 quantiles 参数，用于存储分位数。
        注册 target 缓冲区，用于存储目标值。
        """
        super().__init__(*args, **kwargs)

        self.channels = int(channels)
        self.filters = tuple(int(f) for f in filters)   #* self.filters 的默认值为(3,3,3,3)
        self.init_scale = float(init_scale)
        self.tail_mass = float(tail_mass)

        # Create parameters
        filters = (1,) + self.filters + (1,)    #* 在滤波器列表的前后各添加一个1，表示输入和输出的维度, filters的默认值为(1,3,3,3,3,1)
        scale = self.init_scale ** (1 / (len(self.filters) + 1))    #* 计算每个滤波器的缩放因子scale 为 self.init_scale 的(1 / (len(self.filters) + 1))次方
        channels = self.channels    #* 设置通道数
        
        #! 请参考Balle 2018 论文对应的文档中的 “单变量概率密度模型实现(Univariate density model) / 累计概率密度模型” 部分
        #! 这里的self.matrices[k] 就相当于 f_k, self.biases[k] 就相当于 b_k, self.factors[k] 就相当于 a_k
        self.matrices = nn.ParameterList()  #* 创建权重参数列表self.matrices。
        self.biases = nn.ParameterList()    #* 创建偏置参数列表。
        self.factors = nn.ParameterList()   #* 创建因子参数列表。

        for i in range(len(self.filters) + 1):  #* 遍历每个滤波器。
            init = np.log(np.expm1(1 / scale / filters[i + 1])) #* init 是一个初始值
            matrix = torch.Tensor(channels, filters[i + 1], filters[i]) #* 创建权重张量matrix，维度为 (channels, filters[i + 1], filters[i])。
            matrix.data.fill_(init)     #* 将matrix矩阵填充满初始化值init
            self.matrices.append(nn.Parameter(matrix))  #* 将权重张量matrix添加到参数列表self.matrices中。

            bias = torch.Tensor(channels, filters[i + 1], 1)    #* 创建偏移量张量bias，维度为 (channels, filters[i + 1], 1)。
            nn.init.uniform_(bias, -0.5, 0.5)   #* 将bias矩阵填初始化为(-0.5, 0.5)上的均匀分布
            self.biases.append(nn.Parameter(bias))  #* 将偏置张量bias添加到偏置参数列表self.biases中。

            if i < len(self.filters):   #* 如果当前滤波器不是最后一个，创建因子张量
                factor = torch.Tensor(channels, filters[i + 1], 1)  #* 创建因子张量，维度为 (channels, filters[i + 1], 1)。
                nn.init.zeros_(factor)  #* 使用零初始化因子张量。
                self.factors.append(nn.Parameter(factor))   #* 将因子张量添加到参数列表中

        self.quantiles = nn.Parameter(torch.Tensor(channels, 1, 3)) #* 创建分位数参数张量，这个张量将存储每个通道的三个分位数, 维度为 (channels, 1, 3)。
        init = torch.Tensor([-self.init_scale, 0, self.init_scale]) #* 设置初始化值init = [-self.init_scale, 0, self.init_scale],这些值分别表示分位数的下界、中位数和上界。
        self.quantiles.data = init.repeat(self.quantiles.size(0), 1, 1) #* 重复初始化值以填充 分位数参数张量self.quantiles.data，维度为 (channels, 1, 3)
        #* 其中，self.quantiles.data[i] = [[-self.init_scale, 0, self.init_scale]] (1,3)

        target = np.log(2 / self.tail_mass - 1)
        self.register_buffer("target", torch.Tensor([-target, 0, target]))  #* 注册目标缓冲区，维度为 (3,)。

    def _get_medians(self) -> Tensor:
        """_summary_
        功能
        _get_medians 函数用于从 self.quantiles 参数中提取每个通道的中位数。这些中位数在量化过程中起到重要作用，特别是在定义量化操作的零点附近的行为。

        参数
        无参数
        
        返回值
        返回值：中位数张量，维度为 (channels, 1, 1)。
        """
        medians = self.quantiles[:, :, 1:2] #* 从 self.quantiles 中提取每个通道的中位数medians, 维度为 (channels, 1, 1)。 1:2 表示提取索引为 1 的列，即中位数列。
        #* 假设self.quantiles = tensor([[[-10.,  0., 10.]],[[-10.,  0., 10.]],....])
        #* 那么medians = tensor([[[  0.]],[[  0.]],....])
        return medians

    def update(self, force: bool = False, update_quantiles: bool = False) -> bool:
        """_summary_
        功能：
        update 方法用于更新熵瓶颈层的参数，特别是分位数（quantiles）和量化累积分布函数（CDF）。
        这些参数在训练过程中会根据数据进行调整，以优化量化和编码的性能。该方法确保在编码和解码过程中使用最新的参数，从而提高压缩效率。
        
        参数
        force: bool = False：是否强制更新参数，即使参数已经初始化。
        update_quantiles: bool = False：是否更新分位数。
        
        返回值
        返回值：一个布尔值，表示是否进行了更新。
        
        """
        # Check if we need to update the bottleneck parameters, the offsets are
        # only computed and stored when the conditonal model is update()'d.
        if self._offset.numel() > 0 and not force:
            return False

        if update_quantiles:    #* 如果 update_quantiles 为 True，则调用 _update_quantiles 方法更新分位数。
            self._update_quantiles()
        #* self.quantiles维度为(channels,1,3) self.quantiles[i]=[[最小分位值(下界)，中位数，最大分位值(上界)]] 维度为(1,3)
        medians = self.quantiles[:, 0, 1]   #* medians = self.quantiles[:, 0, 1]：提取每个通道的中位数，维度为 (channels,)。

        minima = medians - self.quantiles[:, 0, 0]  #* 计算每个通道的中位值和最小分位值的距离，维度为 (channels,)
        minima = torch.ceil(minima).int()
        minima = torch.clamp(minima, min=0) #* 限制minma 为非负值

        maxima = self.quantiles[:, 0, 2] - medians  #* 计算每个通道的中位值和最大分位值的距离，维度为 (channels,)。
        maxima = torch.ceil(maxima).int()
        maxima = torch.clamp(maxima, min=0) #* 限制maxima 为非负值
        #* minima 和 maxima 都被向上取整并限制为非负值。

        self._offset = -minima  #* 计算偏移量，维度为 (channels,)。

        pmf_start = medians - minima    #* 计算 PMF 的起始点，维度为 (channels,)，pmf[i] 为当前通道i的起始点，取值为 中位数 - (中位值和最小分位值的距离)
        pmf_length = maxima + minima + 1    #* 计算 PMF 的长度，维度为 (channels,)，pmf_length[i] 表示 当前通道i的pmf的长度

        max_length = pmf_length.max().item()    #* 计算最大长度(pmf_length中的最大长度)，用于生成样本
        device = pmf_start.device
        samples = torch.arange(max_length, device=device)   #* 生成样本samples，维度为 (max_length,), samples[i] = i
        samples = samples[None, :] + pmf_start[:, None, None]   #* 调整样本的维度，维度为 (channels, 1,max_length)
        #* None 表示添加一个维度, samples[None, :] 维度为(1,max_length), pmf_start[:, None, None]维度为: (channels, 1,1)
        #* 由于Python的广播机制，samples变为(channels, 1,max_length), samples[i][1][j] = j
        #* pmf_start变为(channels, 1,max_length), pmf[i][1][j] = 通道i的起点值
        #* 每个通道的 pmf_start 值被加到了 samples 的每个元素上，生成了一个新的张量，其中每个通道的值都偏移了相应的 pmf_start 值
        #* samples[i] 表示 第i个通道的样本， samples[i][1][j] = 原本的pmf_start[i](通道i的起点值) + 原本的samples[j] =  pmf_start[i] + j， 表示第i个通道的PMF的第j个位置的取值
        #* 

        pmf, lower, upper = self._likelihood(samples, stop_gradient=True)   #TODO
        #* pmf, lower, upper 的维度均为(channels, 1,max_length)
        pmf = pmf[:, 0, :]  #* 去掉第二维，pmf维度为(channels,max_length), pmf[i][j] 表示在第i个通道的第j个位置的概率密度函数(pmf)
        tail_mass = torch.sigmoid(lower[:, 0, :1]) + torch.sigmoid(-upper[:, 0, -1:])

        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._cdf_length = pmf_length + 2
        return True

    def loss(self) -> Tensor:
        logits = self._logits_cumulative(self.quantiles, stop_gradient=True)
        loss = torch.abs(logits - self.target).sum()
        return loss

    def _logits_cumulative(self, inputs: Tensor, stop_gradient: bool) -> Tensor:
        """_summary_

        功能
        _logits_cumulative 方法用于计算输入张量的累计概率密度函数CDF的对数值（logits）。
        #! _logits_cumulative 的计算逻辑请参考Balle 2018 论文对应的文档中的 “单变量概率密度模型实现(Univariate density model) / 累计概率密度模型” 部分
        这个方法通过一系列的线性变换和非线性激活函数，将输入张量转换为累积对数。这些累积对数在计算似然和量化过程中起到重要作用。

        参数
        inputs: Tensor：输入张量，维度为 (channels, 1, *)，其中 * 表示任意长度的额外维度。
        stop_gradient: bool：是否停止梯度传播。如果为 True，则在计算过程中不会计算梯度，这在某些优化步骤中很有用。
        
        返回值
        返回值：累积对数张量，维度与输入张量相同,维度为 (channels, 1, *)，其中 * 表示任意长度的额外维度。
        """
        # TorchScript not yet working (nn.Mmodule indexing not supported)
        logits = inputs
        for i in range(len(self.filters) + 1):  #* 遍历每个滤波器，共 len(self.filters) + 1 次
            matrix = self.matrices[i]   #* 获取当前滤波器的权重矩阵，维度为 (channels, filters[i + 1], filters[i])
            if stop_gradient:
                matrix = matrix.detach()
            logits = torch.matmul(F.softplus(matrix), logits)   #* 使用 softplus 激活函数对权重矩阵进行处理，然后与 logits 进行矩阵乘法
            #* logits 当前的维度为: (channels, filters[i],*), matrix 的维度为：(channels, filters[i + 1], filters[i])， 两者矩阵相乘等于: (channels, filters[i + 1], *)

            bias = self.biases[i]   #* 获取当前滤波器的偏置，维度为 (channels, filters[i + 1], 1)
            if stop_gradient:
                bias = bias.detach()
            logits = logits + bias  #* logits 当前的维度为: (channels, filters[i+1],*)

            if i < len(self.filters):   #* 如果当前滤波器不是最后一个，进行非线性激活
                factor = self.factors[i]
                if stop_gradient:
                    factor = factor.detach()
                logits = logits + torch.tanh(factor) * torch.tanh(logits)
        return logits

    def _likelihood(
        self, inputs: Tensor, stop_gradient: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """_summary_

        功能
        _likelihood 方法用于计算输入张量的似然。这个方法通过计算输入值在给定概率分布中的累积概率分布CDF（logits）来确定每个输入值的似然。
        具体来说，它计算每个输入值在量化区间内的概率，这些概率用于后续的熵编码和解码过程。
        likelihood = CDF(input+0.5) - CDF(input-0.5), 表示 input的概率(pmf)
        参数
        inputs: Tensor：输入张量，维度为 ( channels, 1,  *)，其中 * 表示任意长度的额外维度。
        stop_gradient: bool = False：是否停止梯度传播。如果为 True，则在计算过程中不会计算梯度，这在某些优化步骤中很有用。
        
        返回值
        返回值：一个包含三个张量的元组 (likelihood, lower, upper)：
        likelihood：输入值的似然张量，维度与输入张量相同,维度为 ( channels, 1,  *), likelihood = CDF(input+0.5) - CDF(input-0.5), 表示 input的概率(pmf)
        lower：输入值的累积对数下界，维度与输入张量相同,维度为 ( channels, 1,  *)
        upper：输入值的累积对数上界，维度与输入张量相同,维度为 ( channels, 1,  *)

        """
        half = float(0.5)
        lower = self._logits_cumulative(inputs - half, stop_gradient=stop_gradient) #* 计算输入值减去 0.5 后的累积对数。这里 half 是 0.5，用于定义量化区间的边界。
        upper = self._logits_cumulative(inputs + half, stop_gradient=stop_gradient) #* 计算输入值加上 0.5 后的累积对数。
        likelihood = torch.sigmoid(upper) - torch.sigmoid(lower)    #* 使用 sigmoid 函数将累积对数转换为概率，然后计算上界和下界之间的差值，得到输入值的似然(i.e. 概率)。
        return likelihood, lower, upper

    def forward(
        self, x: Tensor, training: Optional[bool] = None
    ) -> Tuple[Tensor, Tensor]:
        """_summary_
        功能
        forward 方法是 EntropyBottleneck 类的前向传播方法，用于处理输入张量 x，并返回量化后的输出张量和似然张量。
        这个方法在训练和推理过程中都会被调用，根据 training 参数的不同，行为也会有所不同。

        参数
        x: Tensor：输入张量，维度为 (batch_size, channels, *)，其中 * 表示任意长度的额外维度。
        training: Optional[bool] = None：是否处于训练模式。如果为 None，则使用 self.training 的值。
        
        返回值
        返回值：一个包含两个张量的元组 (outputs, likelihood)：
        outputs：量化后的输出张量，维度与输入张量相同。
        likelihood：输入值的似然张量，维度与输入张量相同。
        
        实现逻辑：
        
        将x进行一定程度的reshape, 然后将x送进量化器self.quantize进行量化得到量化结果outputs, 
        然后用_likelihood计算似然概率，然后将输出的(量化结果outputs)和(似然概率 likelihood)重新reshape成和输入x相同维度。
        """
        if training is None:    #* 如果 training 为 None，则使用 self.training 的值。
            training = self.training

        if not torch.jit.is_scripting():    #* 如果不在 TorchScript 环境中
            # x from B x C x ... to C x B x ...
            perm = torch.cat(
                (
                    torch.tensor([1, 0], dtype=torch.long, device=x.device),
                    torch.arange(2, x.ndim, dtype=torch.long, device=x.device), #* 生成从x的第2维到最后一维的索引。
                )
            )   #* 生成排列索引perm，将输入张量的维度从 (batch_size, channels, *) 调整为 (channels, batch_size, *)
            inv_perm = perm #* 生成逆排列索引，用于恢复原始维度
        else:
            raise NotImplementedError()
        
        #* 例如: x是一个4维的矩阵，那么x的维度数x.ndim=4, 因此torch.arange(2,x.ndim) = [2,3], 所以perm = (1,0,2,3)
        # TorchScript in 2D for static inference
        # Convert to (channels, ... , batch) format
        # perm = (1, 2, 3, 0)
        # inv_perm = (3, 0, 1, 2)

        x = x.permute(*perm).contiguous()   #* 调整输入张量的维度, 将输入张量x的维度从 (batch_size, channels, *) 调整为 (channels, batch_size, *)
        shape = x.size()    #* 保存x当前的维度shape: (channels, batch_size, *)
        values = x.reshape(x.size(0), 1, -1)    #* 将输入张量展平为 (channels, 1, -1), values的维度为(channels, 1, -1)

        # Add noise or quantize

        outputs = self.quantize(
            values, "noise" if training else "dequantize", self._get_medians()
        )   #* 根据训练模式，对输入张量values 进行量化或反量化。如果 training 为 True，则添加噪声；否则，进行反量化
        #* 量化后的值为 outputs

        if not torch.jit.is_scripting():
            likelihood, _, _ = self._likelihood(outputs)    #* 计算量化后的输出张量的似然
            if self.use_likelihood_bound:
                likelihood = self.likelihood_lower_bound(likelihood)
        else:
            raise NotImplementedError()
            # TorchScript not yet supported
            # likelihood = torch.zeros_like(outputs)

        # Convert back to input tensor shape
        outputs = outputs.reshape(shape)    #* 将量化后的输出张量outputs 恢复为原始形状(channels, batch_size, *)
        outputs = outputs.permute(*inv_perm).contiguous()   #* 将输出张量的维度调整回 (batch_size, channels, *)

        likelihood = likelihood.reshape(shape)  #* 将似然张量恢复为原始形状(channels, batch_size, *)
        likelihood = likelihood.permute(*inv_perm).contiguous() #* 将似然张量的维度调整回 (batch_size, channels, *)

        return outputs, likelihood

    @staticmethod
#* @staticmethod 是一个 Python 装饰器，用于定义静态方法。静态方法与类相关联，但不依赖于类的实例。这意味着静态方法可以在不创建类实例的情况下被调用，并且它们不接收隐式的 self 参数（即类实例）。

#* 静态方法的特点
#* 不依赖于类实例：静态方法不绑定到类的实例，因此它们不接收 self 参数。
#* 可以通过类或实例调用：静态方法可以通过类名或类的实例调用，但通常推荐通过类名调用。
#* 不访问类的属性或方法：静态方法不能访问类的其他属性或方法，因为它们不依赖于类的实例    
    def _build_indexes(size):
        """_summary_
        功能
        _build_indexes 方法用于生成索引张量，这些索引张量用于在熵编码和解码过程中引用每个通道的累积分布函数（CDF）。
        这个方法根据输入张量的大小生成索引张量，确保每个通道的索引在处理过程中被正确引用。

        参数
        size：输入张量的维度大小，类型为 tuple，表示张量的维度，例如 (batch_size, channels, height, width)。
        
        返回值
        返回值：索引张量indexes，维度为 (batch_size, channels, height, width)，其中每个元素是通道索引。
        #! indexes[i][j][m][n]的取值为 j,(对于任意i,m,n)
        
        整体执行逻辑
        确定输入张量的维度：获取输入张量的维度数 dims，批量大小 N 和通道数 C。
        创建索引张量：生成一个从 0 到 C-1 的索引序列，并调整其维度以匹配输入张量的通道维度。
        重复索引张量：将索引张量在批量大小和其他空间维度上重复，以生成最终的索引张量。
        """
        dims = len(size)    #* 获取输入张量的维度数; e.g. dims =4 ， 当x的维度为 (batch_size, channels, height, width)
        N = size[0]     #* 获取批量大小batch_size 
        C = size[1]     #* 获取通道数channel

        view_dims = np.ones((dims,), dtype=np.int64)    #* 创建一个与输入张量维度数相同的数组，初始值全为 1， e.g. view_dims = [1,1,1,1]
        view_dims[1] = -1   #* 将第二个维度（通道维度）设置为 -1，表示该维度将被展平, e.g. view_dims = [1,-1,1,1]
        indexes = torch.arange(C).view(*view_dims)  #* 生成从 0 到 C-1 的索引序列indexes，并调整其维度以匹配输入张量的通道维度，indexes 的维度为: (1, channels, 1, 1)
        indexes = indexes.int()     #* 将indexes的内容设置为int类型

        return indexes.repeat(N, 1, *size[2:])  #* 将索引张量在批量大小和其他空间维度上重复，生成最终的索引张量, 其维度为 (batch_size, channels, height, width)
        #! 其中indexes[i][j][m][n]的取值为 j,(对于任意i,m,n)
        #* 在 PyTorch 中，repeat 函数用于重复张量的元素，以生成一个新的张量。这个函数非常有用，尤其是在需要扩展张量的维度或重复某些模式时。repeat 函数的语法如下：
        # torch.Tensor.repeat(*sizes)
        # 参数：*sizes：一个或多个整数，表示每个维度上重复的次数。
        # 返回值：
        # 返回一个新的张量，其维度根据 sizes 参数扩展。
        #* 例子： x = torch.tensor([[1, 2], [3, 4]])， y = x.repeat(2, 3)
        # y= tensor([[1, 2, 1, 2, 1, 2],
        # [3, 4, 3, 4, 3, 4],
        # [1, 2, 1, 2, 1, 2],
        # [3, 4, 3, 4, 3, 4]])
        
    @staticmethod
    def _extend_ndims(tensor, n):
        return tensor.reshape(-1, *([1] * n)) if n > 0 else tensor.reshape(-1)

    @torch.no_grad()
    def _update_quantiles(self, search_radius=1e5, rtol=1e-4, atol=1e-3):
        """Fast quantile update via bisection search.

        Often faster and much more precise than minimizing aux loss.
        """
        device = self.quantiles.device
        shape = (self.channels, 1, 1)
        low = torch.full(shape, -search_radius, device=device)
        high = torch.full(shape, search_radius, device=device)

        def f(y, self=self):
            return self._logits_cumulative(y, stop_gradient=True)

        for i in range(len(self.target)):
            q_i = self._search_target(f, self.target[i], low, high, rtol, atol)
            self.quantiles[:, :, i] = q_i[:, :, 0]

    @staticmethod
    def _search_target(f, target, low, high, rtol=1e-4, atol=1e-3, strict=False):
        assert (low <= high).all()
        if strict:
            assert ((f(low) <= target) & (target <= f(high))).all()
        else:
            low = torch.where(target <= f(high), low, high)
            high = torch.where(f(low) <= target, high, low)
        while not torch.isclose(low, high, rtol=rtol, atol=atol).all():
            mid = (low + high) / 2
            f_mid = f(mid)
            low = torch.where(f_mid <= target, mid, low)
            high = torch.where(f_mid >= target, mid, high)
        return (low + high) / 2

    def compress(self, x):
        """_summary_
        功能
        compress 方法用于将输入张量 x 压缩为码流字符串。这个方法通过熵编码将输入张量的值转换为紧凑的比特流，适用于图像压缩等应用。
        该方法首先生成索引张量和中位数张量，然后调用基类的 compress 方法进行实际的压缩操作。

        参数
        x: Tensor：输入张量，维度为 (batch_size, channels, height, width)。
        
        返回值
        返回值：压缩后的字符串列表，每个字符串表示一个批次的压缩数据。
        
        整体执行逻辑
        生成索引张量：根据输入张量的大小生成索引张量。
        获取中位数：从 self.quantiles 中提取中位数，并调整其维度以匹配输入张量。
        扩展中位数张量：将中位数张量扩展到与输入张量相同的批量大小和其他空间维度。
        调用基类的 compress 方法：将输入张量、索引张量和中位数张量传递给基类的 compress 方法，进行实际的压缩操作。
        """
        indexes = self._build_indexes(x.size()) #* 调用 _build_indexes 方法生成索引张量indexes，维度为 (batch_size, channels, height, width)
        medians = self._get_medians().detach()  #* 调用 _get_medians 方法获取中位数张量，维度为 (channels, 1, 1)，例如 (128, 1, 1)
        spatial_dims = len(x.size()) - 2    #* 计算空间维度数，例如 2（高度和宽度）
        medians = self._extend_ndims(medians, spatial_dims) #* 调用 _extend_ndims 方法扩展中位数张量的维度，维度为 (channels, 1, 1, 1)
        medians = medians.expand(x.size(0), *([-1] * (spatial_dims + 1)))   #* ：将中位数张量扩展到与输入张量相同的批量大小和其他空间维度，维度为 (batch_size , channels, height, width)
        
        #* 至此， x, indexes, medians 这三个张量的维度都是: (batch_size , channels, height, width)
        #传入父类中的函数将输入张量压缩为码流字符串
        #* symbols[i] 的维度为(channels, height, width), 
        #! indexes[i]的维度为(channels, height, width), 并且indexes[j][m][n]的取值为j
        #* index_list = indexes[i].reshape(-1).int().tolist()的结果为:一个长度为(channels x height x width)的列表，且index_list[j*(heightxwidth) : (j+1)*(heightxwidth)] 的取值为j
        #* 这就表示: symbols[i][j] 中的那 height x width 个元素，都会使用第 j 个CDF(累计概率密度函数)， 也就是说：cdf[j] 是对应于图片第j个通道的累计概率密度函数
        return super().compress(x, indexes, medians)

    def decompress(self, strings, size):
        output_size = (len(strings), self._quantized_cdf.size(0), *size)
        indexes = self._build_indexes(output_size).to(self._quantized_cdf.device)
        medians = self._extend_ndims(self._get_medians().detach(), len(size))
        medians = medians.expand(len(strings), *([-1] * (len(size) + 1)))
        return super().decompress(strings, indexes, medians.dtype, medians)


class GaussianConditional(EntropyModel):
    r"""Gaussian conditional layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.

    This is a re-implementation of the Gaussian conditional layer in
    *tensorflow/compression*. See the `tensorflow documentation
    <https://github.com/tensorflow/compression/blob/v1.3/docs/api_docs/python/tfc/GaussianConditional.md>`__
    for more information.
    """

    def __init__(
        self,
        scale_table: Optional[Union[List, Tuple]],
        *args: Any,
        scale_bound: float = 0.11,
        tail_mass: float = 1e-9,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        if not isinstance(scale_table, (type(None), list, tuple)):
            raise ValueError(f'Invalid type for scale_table "{type(scale_table)}"')

        if isinstance(scale_table, (list, tuple)) and len(scale_table) < 1:
            raise ValueError(f'Invalid scale_table length "{len(scale_table)}"')

        if scale_table and (
            scale_table != sorted(scale_table) or any(s <= 0 for s in scale_table)
        ):
            raise ValueError(f'Invalid scale_table "({scale_table})"')

        self.tail_mass = float(tail_mass)
        if scale_bound is None and scale_table:
            scale_bound = self.scale_table[0]
        if scale_bound <= 0:
            raise ValueError("Invalid parameters")
        self.lower_bound_scale = LowerBound(scale_bound)

        self.register_buffer(
            "scale_table",
            self._prepare_scale_table(scale_table) if scale_table else torch.Tensor(),
        )

        self.register_buffer(
            "scale_bound",
            torch.Tensor([float(scale_bound)]) if scale_bound is not None else None,
        )

    @staticmethod
    def _prepare_scale_table(scale_table):
        return torch.Tensor(tuple(float(s) for s in scale_table))

    def _standardized_cumulative(self, inputs: Tensor) -> Tensor:
        half = float(0.5)
        const = float(-(2**-0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    @staticmethod
    def _standardized_quantile(quantile):
        return scipy.stats.norm.ppf(quantile)

    def update_scale_table(self, scale_table, force=False):
        # Check if we need to update the gaussian conditional parameters, the
        # offsets are only computed and stored when the conditonal model is
        # updated.
        if self._offset.numel() > 0 and not force:
            return False
        device = self.scale_table.device
        self.scale_table = self._prepare_scale_table(scale_table).to(device)
        self.update()
        return True

    def update(self):
        multiplier = -self._standardized_quantile(self.tail_mass / 2)
        pmf_center = torch.ceil(self.scale_table * multiplier).int()
        pmf_length = 2 * pmf_center + 1
        max_length = torch.max(pmf_length).item()

        device = pmf_center.device
        samples = torch.abs(
            torch.arange(max_length, device=device).int() - pmf_center[:, None]
        )
        samples_scale = self.scale_table.unsqueeze(1)
        samples = samples.float()
        samples_scale = samples_scale.float()
        upper = self._standardized_cumulative((0.5 - samples) / samples_scale)
        lower = self._standardized_cumulative((-0.5 - samples) / samples_scale)
        pmf = upper - lower

        tail_mass = 2 * lower[:, :1]

        quantized_cdf = torch.Tensor(len(pmf_length), max_length + 2)
        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._offset = -pmf_center
        self._cdf_length = pmf_length + 2

    def _likelihood(
        self, inputs: Tensor, scales: Tensor, means: Optional[Tensor] = None
    ) -> Tensor:
        half = float(0.5)

        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = self.lower_bound_scale(scales)

        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower

        return likelihood

    def forward(
        self,
        inputs: Tensor,
        scales: Tensor,
        means: Optional[Tensor] = None,
        training: Optional[bool] = None,
    ) -> Tuple[Tensor, Tensor]:
        if training is None:
            training = self.training
        outputs = self.quantize(inputs, "noise" if training else "dequantize", means)
        likelihood = self._likelihood(outputs, scales, means)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        return outputs, likelihood

    def build_indexes(self, scales: Tensor) -> Tensor:
        scales = self.lower_bound_scale(scales)
        indexes = scales.new_full(scales.size(), len(self.scale_table) - 1).int()
        for s in self.scale_table[:-1]:
            indexes -= (scales <= s).int()
        return indexes


class GaussianMixtureConditional(GaussianConditional):
    def __init__(
        self,
        K=3,
        scale_table: Optional[Union[List, Tuple]] = None,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(scale_table, *args, **kwargs)

        self.K = K

    def _likelihood(
        self, inputs: Tensor, scales: Tensor, means: Tensor, weights: Tensor
    ) -> Tensor:
        likelihood = torch.zeros_like(inputs)
        M = inputs.size(1)

        for k in range(self.K):
            likelihood += (
                super()._likelihood(
                    inputs,
                    scales[:, M * k : M * (k + 1)],
                    means[:, M * k : M * (k + 1)],
                )
                * weights[:, M * k : M * (k + 1)]
            )

        return likelihood

    def forward(
        self,
        inputs: Tensor,
        scales: Tensor,
        means: Tensor,
        weights: Tensor,
        training: Optional[bool] = None,
    ) -> Tuple[Tensor, Tensor]:
        if training is None:
            training = self.training
        outputs = self.quantize(
            inputs, "noise" if training else "dequantize", means=None
        )
        likelihood = self._likelihood(outputs, scales, means, weights)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        return outputs, likelihood

    @torch.no_grad()
    def _build_cdf(self, scales, means, weights, abs_max):
        #这个函数将概率分布转换为可以用于编码的CDF
        num_latents = scales.size(1)
        num_samples = abs_max * 2 + 1
        TINY = 1e-10
        device = scales.device

        scales = scales.clamp_(0.11, 256)
        means += abs_max

        scales_ = scales.unsqueeze(-1).expand(-1, -1, num_samples)
        means_ = means.unsqueeze(-1).expand(-1, -1, num_samples)
        weights_ = weights.unsqueeze(-1).expand(-1, -1, num_samples)

        samples = (
            torch.arange(num_samples).to(device).unsqueeze(0).expand(num_latents, -1)
        )

        pmf = torch.zeros_like(samples).float()
        for k in range(self.K):
            pmf += (
                0.5
                * (
                    1
                    + torch.erf(
                        (samples + 0.5 - means_[k]) / ((scales_[k] + TINY) * 2**0.5)
                    )
                )
                - 0.5
                * (
                    1
                    + torch.erf(
                        (samples - 0.5 - means_[k]) / ((scales_[k] + TINY) * 2**0.5)
                    )
                )
            ) * weights_[k]

        cdf_limit = 2**self.entropy_coder_precision - 1
        pmf = torch.clamp(pmf, min=1.0 / cdf_limit, max=1.0)
        pmf_scaled = torch.round(pmf * cdf_limit)
        pmf_sum = torch.sum(pmf_scaled, 1, keepdim=True).expand(-1, num_samples)

        cdf = F.pad(
            torch.cumsum(pmf_scaled * cdf_limit / pmf_sum, 1).int(),
            (1, 0),
            "constant",
            0,
        )
        pmf_quantized = torch.diff(cdf, dim=1)

        # We can't have zeros in PMF because rANS won't be able to encode it.
        # Try to fix this by "stealing" probability from some unlikely symbols.

        pmf_zero_count = num_samples - torch.count_nonzero(pmf_quantized, dim=1)

        _, pmf_first_stealable_indices = torch.min(
            torch.where(
                pmf_quantized > pmf_zero_count.unsqueeze(-1).expand(-1, num_samples),
                pmf_quantized,
                torch.tensor(cdf_limit + 1).int(),
            ),
            dim=1,
        )

        pmf_real_zero_indices = (pmf_quantized == 0).nonzero().transpose(0, 1)
        pmf_quantized[pmf_real_zero_indices[0], pmf_real_zero_indices[1]] += 1

        pmf_real_steal_indices = torch.cat(
            (
                torch.arange(num_latents).to(device).unsqueeze(-1),
                pmf_first_stealable_indices.unsqueeze(-1),
            ),
            dim=1,
        ).transpose(0, 1)
        pmf_quantized[
            pmf_real_steal_indices[0], pmf_real_steal_indices[1]
        ] -= pmf_zero_count

        cdf = F.pad(torch.cumsum(pmf_quantized, 1).int(), (1, 0), "constant", 0)
        cdf = F.pad(cdf, (0, 1), "constant", cdf_limit + 1)

        return cdf

    def reshape_entropy_parameters(self, scales, means, weights, nonzero):
        reshape_size = (scales.size(0), self.K, scales.size(1) // self.K, -1)

        scales = (
            scales.reshape(*reshape_size)[:, :, nonzero]
            .permute(1, 0, 2, 3)
            .reshape(self.K, -1)
        )
        means = (
            means.reshape(*reshape_size)[:, :, nonzero]
            .permute(1, 0, 2, 3)
            .reshape(self.K, -1)
        )
        weights = (
            weights.reshape(*reshape_size)[:, :, nonzero]
            .permute(1, 0, 2, 3)
            .reshape(self.K, -1)
        )
        return scales, means, weights

    def compress(self, y, scales, means, weights):
        abs_max = (
            max(torch.abs(y.max()).int().item(), torch.abs(y.min()).int().item()) + 1
        )
        abs_max = 1 if abs_max < 1 else abs_max

        y_quantized = torch.round(y)
        zero_bitmap = torch.where(
            torch.sum(torch.abs(y_quantized), (3, 2)).squeeze(0) == 0, 0, 1
        )

        nonzero = torch.nonzero(zero_bitmap).flatten().tolist()
        symbols = y_quantized[:, nonzero] + abs_max
        cdf = self._build_cdf(
            *self.reshape_entropy_parameters(scales, means, weights, nonzero), abs_max
        )

        num_latents = cdf.size(0)

        # rv = self.entropy_coder._encoder.encode_with_indexes(
        #     symbols.reshape(-1).int().tolist(),
        #     torch.arange(num_latents).int().tolist(),
        #     cdf.cpu().to(torch.int32),
        #     torch.tensor(cdf.size(1)).repeat(num_latents).int().tolist(),
        #     torch.tensor(0).repeat(num_latents).int().tolist(),
        # )
        rv = self.entropy_coder._encoder.encode_with_indexes(
            symbols.reshape(-1).int().tolist(),
            torch.arange(num_latents).int().tolist(),
            cdf.cpu().tolist(),
            torch.tensor(cdf.size(1)).repeat(num_latents).int().tolist(),
            torch.tensor(0).repeat(num_latents).int().tolist(),
        )

        return (rv, abs_max, zero_bitmap), y_quantized

    def decompress(self, strings, abs_max, zero_bitmap, scales, means, weights):
        nonzero = torch.nonzero(zero_bitmap).flatten().tolist()
        cdf = self._build_cdf(
            *self.reshape_entropy_parameters(scales, means, weights, nonzero), abs_max
        )

        num_latents = cdf.size(0)

        # values = self.entropy_coder._decoder.decode_with_indexes(
        #     strings,
        #     torch.arange(num_latents).int().tolist(),
        #     cdf.cpu().to(torch.int32),
        #     torch.tensor(cdf.size(1)).repeat(num_latents).int().tolist(),
        #     torch.tensor(0).repeat(num_latents).int().tolist(),
        # )
        values = self.entropy_coder._decoder.decode_with_indexes(
            strings,
            torch.arange(num_latents).int().tolist(),
            cdf.cpu().tolist(),
            torch.tensor(cdf.size(1)).repeat(num_latents).int().tolist(),
            torch.tensor(0).repeat(num_latents).int().tolist(),
        )

        symbols = torch.tensor(values) - abs_max
        symbols = symbols.reshape(scales.size(0), -1, scales.size(2), scales.size(3))

        y_hat = torch.zeros(
            scales.size(0), zero_bitmap.size(0), scales.size(2), scales.size(3)
        )
        y_hat[:, nonzero] = symbols.float()

        return y_hat
