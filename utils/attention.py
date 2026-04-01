import math

import torch
from torch import nn
class se_block(nn.Module):
    # channel 输入进来的通道数
    # ratio 代表缩放的比例（第一次全连接会用到） 因为第一次全连接 神经元的个数较少 所以我们会进行一个缩放
    def __init__(self,channel,ratio = 16):
        # 初始化
        super(se_block, self).__init__()
        # 在高和宽上进行自适应全局平均池化 池化完成之后 高和宽就全部都是1了
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 两次全连接
        # 先定义神经元个数较少的全连接
        self.fc = nn.Sequential(
            # 第一次全连接
            # 该函数中参数in_features代表输入的神经元个数，值为channel
            # 参数out_features代表输出的神经元个数，值为channel // ratio，即会进行我们神经元个数的减少
            # False 代表不使用偏置量
            nn.linear(channel, channel // ratio, False),
            # ReLU激活函数
            nn.ReLU(),
            # 第二次全连接
            nn.Linear(channel // ratio, channel, False),
            # Sigmoid激活函数，把值固定到0与1之间
            nn.Sigmoid(),
        )

    # 前向传播
    def forward(self, x):
        # 首先计算一下输入进来的特征层的size
        # 第一维度为batch_size，第二维度为通道数，第三维度为高，第四维度为宽
        b, c, h, w = x.size()
        # 先对输入进来的特征层进行全局平均池化
        # 全局平均池化之后，输入进来的shape会变 b, c, h, w -> b, c, 1, 1
        # 但是我们要用view把后两维去掉 b, c, 1, 1 -> b, c
        avg = self.avg_pool(x).view([b, c])
        # 然后进行两次全连接，然后用view对全连接后的结果再次进行一个reshape，方便后续的处理
        # 变化过程 b,c -> b, c//ratio -> b,c -> b,c,1,1
        fc = self.fc(avg).view([b, c, 1, 1])
        # 看一下每一个通道的权值
        # print(fc)
        # 将两次全连接后的结果 乘上输入进来的特征层
        return x * fc


class channel_attention(nn.Module):
    def __init__(self, channel, ratio = 16):
        super(channel_attention, self).__init__()
        # 对输入进来的特征层进行一个自适应最大池化，输出的特征层的高和宽是1
        self.max_pool = nn.AdaptiveAvgPool2d(1)
        # 对输入进来的特征层进行一个平均池化，输出的特征层的高和宽是1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 定义两次全连接
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, False),
            # 由于是相加之后才取sigmoid的，所以此处尚未取sigmoid
        )
        # 定义sigmoid函数
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, x):
        b, c, h, w = x.size()
        # 对输入进来的特征层x 取一个全局最大池化和全局平均池化
        max_pool_out = self.max_pool(x).view([b, c])
        avg_pool_out = self.avg_pool(x).view([b, c])
        # 对这两次池化后的结果 使用共享的全连接层进行处理
        max_fc_out = self.fc(max_pool_out)
        avg_fc_out = self.fc(avg_pool_out)
        # 将上述两个结果进行相加
        out = max_fc_out + avg_fc_out
        # 将相加后的结果再取sigmoid，这样处理相当于获得输入进来的特征层，每一个通道的比重
        # 用view进行reshape，把它的高宽维度补回去
        out = self.sigmoid(out).view([b, c, 1, 1])
        return out * x

# 空间注意力机制
class spacial_attention(nn.Module):
    # 对于空间注意力机制 我们不需要关注通道数 所以参数中不再需要传入通道数
    # 但是，它中间会进行一次卷积，所以我们会关注它的卷积核大小kernel_size，其值一般为3或7
    def __init__(self, kernel_size = 7):
        super(spacial_attention, self).__init__()
        # 因为设置的卷积核大小为7，所以padding为3
        padding = 7 // 2
        # 定义卷积 输入通道数是2，输出通道数是1，卷积核大小默认设置成7，步长为1（因为不需要压缩特征层的高宽）
        self.conv = nn.Conv2d(2, 1, kernel_size, 1, padding, bias = False)
        # 定义sigmoid函数
        self.sigmoid = nn.Sigmoid()

    # 定义前向传播网络
    def forward(self, x):
        b, c, h, w = x.size()
        # 对特征层再通道上进行最大池化和平均池化
        # 因为对于pythoch来讲 其通道数在第1维度 即batch_size后面的维度 所以dim=1
        # 我们需要把通道这一维度保留下来，所以设置keepdim = 1
        max_pool_out, _ = torch.max(x, dim=1, keepdim=True)
        mean_pool_out = torch.mean(x, dim=1, keepdim=True)
        # 将这两个结果在第一维度，即：通道，进行堆叠
        pool_out = torch.cat([max_pool_out, mean_pool_out], dim=1)
        # 将堆叠后的结果取一个卷积，再取一个sigmoid
        out = self.conv(pool_out)
        out = self.sigmoid(out)
        return out * x

# 将空间注意力机制与通道注意力机制进行结合
class cbam_block(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=7):
        super(cbam_block, self).__init__()
        # 先定义通道注意力机制
        self.channel_attention = channel_attention(channel, ratio)
        # 再定义空间注意力机制
        self.spacial_attention = spacial_attention(kernel_size)

    # 定义前向传播网络
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spacial_attention(x)
        return x

class eca_block(nn.Module):
    # 对于ECANet来说，需要根据通道数计算其卷积核大小
    def __init__(self, channel, gamma=2, b=1):
        super(eca_block, self).__init__()
        # 根据通道数自适应卷积核大小 输入通道数较大时，卷积核就会更大一点，否则就小一点
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        padding = kernel_size // 2
        # 对输入进来的特征层进行全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 定义1D卷积
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 计算输入进来的特征层的batch_size，通道数，高和宽
        b, c, h, w = x.size()
        # 对输入进来的特征层进行全局平均池化，然后进行一个reshape，调整成序列的形式，这样便把时序调整完成了
        # 第一个维度为batch_size，第二个维度代表每一个step的特征长度，第三个维度代表每一个时序
        avg = self.avg_pool(x).view([b, 1, c])
        # 对调整后的结果进行1D卷积
        out = self.conv(avg)
        # 取sigmoid, 获得每一个通道的权值，然后对此结果再次进行一个reshape，方便后续处理
        out = self.sigmoid(out).view([b, c, 1, 1])
        return out * x

# CBAM模块
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        # 通道注意力模块
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, 1, 0),
            nn.Sigmoid()
        )

        # 空间注意力模块
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, 1, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        channel_attention = self.channel_attention(x)
        x = x * channel_attention

        # 空间注意力
        avg_pool = torch.mean(x, 1, keepdim=True)
        max_pool, _ = torch.max(x, 1, keepdim=True)
        spatial_attention = self.spatial_attention(torch.cat([avg_pool, max_pool], 1))
        x = x * spatial_attention

        return x