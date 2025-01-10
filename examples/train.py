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


#* train.py 文件 是一个用于训练图像压缩模型的Python脚本。它使用了PyTorch框架和compressai库，提供了完整的训练流程，包括数据加载、模型训练、验证和保存。

import argparse
import random
import shutil
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.losses import RateDistortionLoss
from compressai.optimizers import net_aux_optimizer
from compressai.zoo import image_models


class AverageMeter:
    """
    Compute running average.
    功能：计算运行平均值。
    
    方法：
    __init__：初始化实例，设置初始值为0。
    update：更新平均值，传入新值和样本数量。    
    
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """
    Custom DataParallel to access the module methods.
    
    功能：自定义的DataParallel类(数据并行处理类)，用于访问模块方法。
    
    方法：
    __getattr__：重载属性访问方法，尝试从父类获取属性，如果失败则从模块中获取。    
    
    """

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """
    Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers
    
    功能：配置主优化器和辅助优化器。
    
    参数：
    net：神经网络模型。
    args：命令行参数。
    
    返回值：主优化器(用于优化网络模型参数)和辅助优化器(用于优化熵模型参数)。
    
    实现：根据配置创建优化器。
    
    """
    conf = {
        "net": {"type": "Adam", "lr": args.learning_rate},
        "aux": {"type": "Adam", "lr": args.aux_learning_rate},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm
):
    """_summary_
    功能：训练模型一个epoch。
    
    参数：
    model：神经网络模型。
    criterion：损失函数。
    train_dataloader：训练数据加载器。
    optimizer：主优化器。(用于优化网络模型参数)
    aux_optimizer：辅助优化器。(用于优化熵模型参数)
    epoch：当前epoch。
    clip_max_norm：梯度裁剪的最大范数。
    
    实现：遍历训练数据，计算损失，进行反向传播和优化器更新。

    """
    model.train()   #* 将模型设置为训练模式，需要计算梯度.
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):    
        #* 遍历训练数据加载器，每次遍历一个小批量(batch), d是一个小批量的数据
        d = d.to(device)

        optimizer.zero_grad()   #* 主优化器梯度清零
        aux_optimizer.zero_grad()   #* 辅助优化器梯度清零

        out_net = model(d)  #* 将数据d输入给模型model, 让模型进行前向传播，计算梯度； 模型的输出为out_net

        out_criterion = criterion(out_net, d)   #* 计算原始输入d 和 模型输出out_net 之间的损失
        out_criterion["loss"].backward()    #* 梯度反向传播
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()    #* 更新模型参数

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 10 == 0:
            #* 每隔 10 个batch，就打印一些信息
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )


def test_epoch(epoch, test_dataloader, model, criterion):
    """_summary_

    功能：测试模型一个epoch。
    
    参数：
    epoch：当前epoch数目。
    test_dataloader：测试数据加载器。
    model：神经网络模型。
    criterion：损失函数。
    
    返回值：平均损失。
    
    实现：遍历测试数据，计算损失，更新平均损失
    """
    model.eval()    #* 设置模型为测试状态，不要计算梯度
    device = next(model.parameters()).device

    loss = AverageMeter()    #* 定义一个损失值指标统计器，用于统计平均损失值
    bpp_loss = AverageMeter()    #* 定义一个bpp指标统计器，用于统计平均bpp
    mse_loss = AverageMeter()    #* 定义一个mse指标统计器，用于统计平均mse
    aux_loss = AverageMeter()    #* 定义一个aux指标统计器，用于统计平均aux(平均辅助损失)

    with torch.no_grad():
        for d in test_dataloader:
            #* 遍历 测试数据加载器test_dataloader中的每一组数据，d 表示一组小批量数据
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

    #* 遍历完测试数据后，打印出统计信息
    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    return loss.avg


def save_checkpoint(state, is_best, filename="results/myresult/checkpoint.pth.tar"):
    """_summary_
    功能：保存模型检查点。
    tar文件可以理解成一个“文件夹的压缩包”，里面存放了多个.pth文件，每个.pth文件就是模型在某一个epoch轮下的参数文件。
    参数：
    state：模型状态字典。
    is_best：是否为最佳模型。
    filename：保存文件名。
    
    实现：保存模型状态，如果为最佳模型则复制到最佳模型文件。
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "results/myresult/checkpoint_best_loss.pth.tar")


def parse_args(argv):
    """_summary_
    功能：解析命令行参数。
    
    参数：命令行参数列表。
    
    返回值：解析后的参数。
    
    实现：使用argparse定义和解析命令行参数。
    """
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="bmshj2018-factorized",
        choices=image_models.keys(),
        help="Model architecture (default: %(default)s)",
    )   #* 解析器parser添加一个可选参数 model, 用于指定要训练的模型
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )   #* 解析器parser添加一个可选参数 datasets(这个参数必须要提供), 用于指定要使用的训练数据集
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )   #* 解析器parser添加一个可选参数 epochs, 用于指定要训练的轮数
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )   #* 解析器parser添加一个可选参数 learning-rate, 用于指定模型的初始学习率
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        type=float,
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )   #* 解析器parser添加一个可选参数 patch-size, 用于指定输入图片的尺寸，将输入图像通过transform变换到patch-size 这个尺寸
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument("--seed", type=int, help="Set random seed for reproducibility")   #* 解析器parser添加一个可选参数 seed, 用于指定随机数种子
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)     #* 调用parse_args函数解析命令行参数。

    if args.seed is not None:   #* 如果指定了随机种子，则设置随机种子以确保结果可复现。
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )   #* 定义训练数据的转换操作，包括随机裁剪和转换为张量

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )   #* 定义测试数据的转换操作，包括随机裁剪和转换为张量

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)    #* 使用ImageFolder加载训练数据集。
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)   #* 使用ImageFolder加载测试数据集。

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"   

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )   #* 创建训练数据加载器。

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )   #* 创建测试数据加载器。

    net = image_models[args.model](quality=3)   #* 根据命令行参数选择模型架构
    net = net.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)  #* 调用configure_optimizers函数配置主优化器和辅助优化器
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateDistortionLoss(lmbda=args.lmbda)    #* 定义模型的损失函数为率失真损失函数

    last_epoch = 0  #* 上一次训练到的epoch轮数
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])








    best_loss = float("inf")    #* best_loss：初始化最佳损失为无穷大
    for epoch in range(last_epoch, args.epochs):    #* 遍历每个epoch，从 last_epoch 到 args.epochs：
        #* 每一个epoch，让模型在整个训练数据集上训练一轮，打印出训练性能;
        #* 然后在整个测试数据集上测试性能
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")  #* 打印当前学习率。
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
        )   #* 调用 train_one_epoch 函数训练模型一个epoch。
        loss = test_epoch(epoch, test_dataloader, net, criterion)   #* 调用 test_epoch 函数测试模型一个epoch，返回平均损失。
        lr_scheduler.step(loss) #* 更新学习率调度器。

        is_best = loss < best_loss  #* 检查是否为最佳模型，is_best 用来表示是否为最佳模型
        best_loss = min(loss, best_loss)    #* 更新 best_loss。

        if args.save:   #* 如果 args.save 为 True，则在每一个epoch轮结束后保存模型检查点。
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
            )


if __name__ == "__main__":
    main(sys.argv[1:])
