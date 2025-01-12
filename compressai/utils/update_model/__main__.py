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

"""
在update_model这个文件夹下，有__main__.py文件，因此可以直接用python命令执行这个文件件(模块)，
e.g. python update_model
这样子的命令会自动执行update_model模块中的__main__.py文件中的main函数

__main__.py 代码是一个 Python 脚本，用于更新 训练好的模型的累积分布函数（CDFs）参数，
模型在训练好之后，我们可以用这个update_model模块来更新模型的参数
并为模型文件添加哈希前缀，以便可以通过 load_state_dict_from_url 函数加载。


Update the CDFs parameters of a trained model.

To be called on a model checkpoint after training. This will update the internal
CDFs related buffers required for entropy coding.
"""
import argparse
import hashlib
import sys

from pathlib import Path
from typing import Dict

import torch

from compressai.models.google import (
    FactorizedPrior,
    JointAutoregressiveHierarchicalPriors,
    MeanScaleHyperprior,
    ScaleHyperprior,
)
from compressai.models.video.google import ScaleSpaceFlow
from compressai.zoo import load_state_dict
from compressai.zoo.image import model_architectures as image_architectures
from compressai.zoo.image_vbr import model_architectures as image_architectures_vbr


def sha256_file(filepath: Path, len_hash_prefix: int = 8) -> str:
    # from pytorch github repo
    sha256 = hashlib.sha256()
    with filepath.open("rb") as f:
        while True:
            buf = f.read(8192)
            if len(buf) == 0:
                break
            sha256.update(buf)
    digest = sha256.hexdigest()

    return digest[:len_hash_prefix]


def load_checkpoint(filepath: Path, arch: str) -> Dict[str, torch.Tensor]:
    """
    功能
    load_checkpoint 函数用于加载模型检查点文件，并返回模型的状态字典。这个函数处理不同格式的检查点文件，确保可以正确加载模型的状态字典。

    参数
    filepath: Path：检查点文件的路径。
    arch: str：模型架构名称。
    
    返回值
    返回值：模型的状态字典，类型为 Dict[str, torch.Tensor]。
    
    整体执行逻辑
    加载检查点文件：使用 torch.load 加载检查点文件。
    提取状态字典：根据检查点文件的格式，提取状态字典。
    调用 load_state_dict 函数加载状态字典。
    返回状态字典：返回处理后的状态字典。
    
    
    """
    checkpoint = torch.load(filepath, map_location="cpu")

    if "network" in checkpoint:
        state_dict = checkpoint["network"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    # if arch in ["bmshj2018-hyperprior-vbr", "mbt2018-mean-vbr"]:
    #     state_dict = load_state_dict(state_dict, vr_entbttlnck=True)
    # else:
    state_dict = load_state_dict(state_dict)
    return state_dict


description = """
Export a trained model to a new checkpoint with updated CDFs and a
hash prefix so that it can be loaded later via `load_state_dict_from_url`.
""".strip()

models = {
    "factorized-prior": FactorizedPrior,
    "jarhp": JointAutoregressiveHierarchicalPriors,
    "mean-scale-hyperprior": MeanScaleHyperprior,
    "scale-hyperprior": ScaleHyperprior,
    "ssf2020": ScaleSpaceFlow,
}
models.update(image_architectures)
models.update(image_architectures_vbr)


def setup_args():
    """_summary_
    功能： 
    为update_model 模块 设置 命令行参数解析器
    
    
    
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "filepath", type=str, help="Path to the checkpoint model to be exported."
    )   #* 添加 位置参数 filepath, 表示
    parser.add_argument("-n", "--name", type=str, help="Exported model name.")
    parser.add_argument("-d", "--dir", type=str, help="Exported model directory.")
    parser.add_argument(
        "--no-update",
        action="store_true",
        default=False,
        help="Do not update the model CDFs parameters.",
    )
    parser.add_argument(
        "-a",
        "--architecture",
        default="scale-hyperprior",
        choices=models.keys(),
        help="Set model architecture (default: %(default)s).",
    )
    return parser


def main(argv):
    """_summary_
    功能
    main 函数是脚本的入口点，用于更新训练好的模型的累积分布函数（CDFs）参数，并为模型文件添加哈希前缀，以便可以通过 load_state_dict_from_url 函数加载。
    这个函数处理命令行参数，加载模型检查点，更新模型参数，保存新的检查点文件，并重命名文件以包含哈希前缀。

    参数
    argv：命令行参数列表，通常为 sys.argv[1:]。
    
    返回值
    返回值：无返回值，但会生成并保存新的模型检查点文件。
    
    整体执行逻辑
    解析命令行参数：使用 setup_args 解析命令行参数。
    加载检查点文件：使用 load_checkpoint 加载模型检查点文件。
    创建模型实例：根据指定的架构创建模型实例。
    更新模型参数：如果 --no-update 未设置，则更新模型的 CDFs 参数。
    保存新的检查点文件：保存更新后的模型状态字典。
    计算文件哈希值：计算新文件的 SHA-256 哈希值前缀。
    重命名文件：将新文件重命名为包含哈希前缀的名称。
    """
    args = setup_args().parse_args(argv)    #* 解析传给update_model 模块的命令行参数。

    filepath = Path(args.filepath).resolve()    #* 解析并规范化检查点文件路径 filepath
    if not filepath.is_file():  #* 检查文件是否存在。
        raise RuntimeError(f'"{filepath}" is not a valid file.')

    state_dict = load_checkpoint(filepath, args.architecture)   #* 加载模型检查点文件并返回状态字典

    model_cls_or_entrypoint = models[args.architecture]
    if not isinstance(model_cls_or_entrypoint, type):
        model_cls = model_cls_or_entrypoint()
    else:
        model_cls = model_cls_or_entrypoint
    if args.architecture in ["bmshj2018-hyperprior-vbr", "mbt2018-mean-vbr"]:
        #* 如果架构是 bmshj2018-hyperprior-vbr 或 mbt2018-mean-vbr，则调用 from_state_dict 方法时传入 vr_entbttlnck=True
        net = model_cls.from_state_dict(state_dict, vr_entbttlnck=True)
    else:
        net = model_cls.from_state_dict(state_dict) #* 使用状态字典state_dict 创建模型实例 net。

    if not args.no_update:  #* 如果 --no-update 未设置，则更新模型的 CDFs 参数
        net.update(force=True)
    state_dict = net.state_dict()

    if not args.name:
        filename = filepath
        while filename.suffixes:
            filename = Path(filename.stem)
    else:
        filename = args.name

    ext = "".join(filepath.suffixes)

    if args.dir is not None:
        output_dir = Path(args.dir)
        Path(output_dir).mkdir(exist_ok=True)
    else:
        output_dir = Path.cwd()

    filepath = output_dir / f"{filename}{ext}"
    torch.save(state_dict, filepath)
    hash_prefix = sha256_file(filepath)

    filepath.rename(f"{output_dir}/{filename}-{hash_prefix}{ext}")


if __name__ == "__main__":
    main(sys.argv[1:])
