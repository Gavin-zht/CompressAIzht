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

import torch
import torch.nn as nn

from torch import Tensor


def lower_bound_fwd(x: Tensor, bound: Tensor) -> Tensor:
    return torch.max(x, bound)


def lower_bound_bwd(x: Tensor, bound: Tensor, grad_output: Tensor):
    pass_through_if = (x >= bound) | (grad_output < 0)
    return pass_through_if * grad_output, None


class LowerBoundFunction(torch.autograd.Function):
    """Autograd function for the `LowerBound` operator.
    功能：
    LowerBoundFunction 是一个自定义的 PyTorch 自动梯度函数，用于实现下界操作（LowerBound）。这个函数的主要作用是确保输入张量 x 的值不会低于某个指定的下界 bound。
    在前向传播中，它会将 x 和 bound 中的较大值返回；
    在反向传播中，它会根据条件传递梯度
    
    2. 参数分析
    x: Tensor：输入张量，可以是任意形状的 PyTorch 张量。
    bound: Tensor：下界张量，与 x 的形状相同，用于指定每个位置的下界值。
    
    3. 返回值分析
    前向传播：返回 torch.max(x, bound)，即 x 和 bound 中的逐元素最大值。
    反向传播：返回两个值：
    第一个值是传递给输入 x 的梯度。
    第二个值是传递给 bound 的梯度（在这个实现中为 None）。
    
    4. 整体执行逻辑
    LowerBoundFunction 的执行逻辑分为两部分：
    前向传播：调用 lower_bound_fwd 函数，计算 x 和 bound 的逐元素最大值。
    反向传播：调用 lower_bound_bwd 函数，根据条件传递梯度。    
    
    
    
    """


    @staticmethod
    def forward(ctx, x, bound):
        """_summary_
        forward 是一个静态方法，无需创建LowerBoundFunction类的实例就可以直接调用
        功能：对输入x取下界函数(i.e. )
        
        """
        ctx.save_for_backward(x, bound)
        return lower_bound_fwd(x, bound)

    @staticmethod
    def backward(ctx, grad_output):
        """_summary_
        forward 是一个静态方法，无需创建LowerBoundFunction类的实例就可以直接调用
        """
        x, bound = ctx.saved_tensors
        return lower_bound_bwd(x, bound, grad_output)


class LowerBound(nn.Module):
    """Lower bound operator, computes `torch.max(x, bound)` with a custom
    gradient.

    The derivative is replaced by the identity function when `x` is moved
    towards the `bound`, otherwise the gradient is kept to zero.
    
    1. 功能概述
    LowerBound 是一个 PyTorch 模块，用于实现一个自定义的下界操作。它的主要功能是确保输入张量 x 的值不会低于某个指定的下界 bound。
    在前向传播中，它会计算 torch.max(x, bound)，而在反向传播中，它会根据自定义的梯度规则传递梯度。
    
    2. 参数分析
    bound: float：下界值，初始化时传入的标量值，用于指定所有元素的下界。
    bound: Tensor：在模块中注册的张量，存储了下界值。
    
    3. 返回值分析
    前向传播：返回 torch.max(x, bound)，即 x 和 bound 中的逐元素最大值。
    反向传播：根据自定义的梯度规则传递梯度。如果 x 大于等于 bound 或者梯度为负，则将梯度传递给 x，否则梯度为零。
    
    4. 整体执行逻辑
    LowerBound 的执行逻辑分为两部分：
    前向传播：调用 LowerBoundFunction.apply 或 torch.max，具体取决于是否在 TorchScript 环境中运行。
    反向传播：通过 LowerBoundFunction 的自定义梯度规则传递梯度。    
    
    
    """

    bound: Tensor

    def __init__(self, bound: float):
        super().__init__()
        self.register_buffer("bound", torch.Tensor([float(bound)]))

    @torch.jit.unused
    def lower_bound(self, x):
        return LowerBoundFunction.apply(x, self.bound)

    def forward(self, x):
        """_summary_
        变量维度：
        x：输入张量，可以是任意形状的 PyTorch 张量，例如 [batch_size, channels, height, width]。
        self.bound：下界张量，形状为 [1]，存储了标量 bound 的值。
        
        逻辑：
        如果在 TorchScript 环境中运行（torch.jit.is_scripting() 为 True），直接使用 torch.max 计算逐元素最大值。
        否则，调用 self.lower_bound 方法，使用自定义的 LowerBoundFunction。
        """
        if torch.jit.is_scripting():
            return torch.max(x, self.bound)
        return self.lower_bound(x)
