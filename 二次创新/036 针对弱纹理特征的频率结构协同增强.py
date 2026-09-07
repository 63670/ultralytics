import torch
import torch.nn as nn
import torch.nn.functional as F

""" 
   针对弱纹理特征的频率结构协同增强模块：
        写作思路与代码讲解：https://www.bilibili.com/video/BV1DuVB6oEJn/
        作用位置：任何单一特征处理时/任何普通卷积，或者任何即插即用模块中。
        主要功能（写作要点）：①通过多分支提取与目标、边界、纹理的关键线索，缓解普通卷积只做局部采样、难以区分有效结构与背景响应的问题；
        ②基于门控的竞争融合将把有价值分支稳定注入原始特征；③突出弱纹理目标中的局部突变与边界响应，降低平滑背景对特征的干扰。（将在本视频的写作部分展开阐述）
        代码层面：首先进入低频结构、高频细节、边缘纹理和背景抑制分支，分别描述局部细节、结构上下文和显著区域；随后生成融合权重，
                    将增强信息重新作用到原始特征上。相比固定相加或单一注意力，该流程更强调内容自适应和任务相关特征的强化。
"""

# 基础卷积模块：Conv2d + GroupNorm + SiLU，用于统一特征变换
class ConvGNAct(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, groups=1, act=True, dilation=1):
        # 调用父类初始化函数，初始化 nn.Module 的内部机制
        super().__init__()

        # 根据卷积核大小和空洞率自动计算 padding，保证输入输出空间尺寸一致
        padding = dilation * (kernel_size // 2)

        # 构建卷积、归一化和激活函数组成的基础处理单元
        self.block = nn.Sequential(
            # 使用二维卷积提取局部特征或调整通道数
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False
            ),

            # 使用 GroupNorm 对输出特征进行归一化，稳定训练过程
            nn.GroupNorm(1, out_channels),

            # 如果 act=True，则使用 SiLU 激活函数；否则不进行激活
            nn.SiLU(inplace=True) if act else nn.Identity()
        )

    def forward(self, x):
        # 将输入特征送入卷积归一化激活模块
        out = self.block(x)

        # 返回处理后的特征
        return out


# 深度可分离卷积模块：先逐通道提取空间信息，再用 1×1 卷积融合通道信息
class DepthwiseSeparableConv(nn.Module):

    def __init__(self, dim, kernel_size=3, dilation=1):
        # 调用父类初始化函数
        super().__init__()

        # 根据卷积核大小和空洞率计算 padding，保持特征图大小不变
        padding = dilation * (kernel_size // 2)

        # 深度卷积：每个通道单独卷积，用于低成本提取空间纹理特征
        self.depthwise = nn.Conv2d(
            dim,
            dim,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=dim,
            bias=False
        )

        # 逐点卷积：使用 1×1 卷积融合不同通道之间的信息
        self.pointwise = nn.Conv2d(
            dim,
            dim,
            kernel_size=1,
            bias=False
        )

        # 使用 GroupNorm 对卷积后的特征进行归一化
        self.norm = nn.GroupNorm(1, dim)

        # 使用 SiLU 激活函数增强非线性表达能力
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        # 先通过深度卷积提取每个通道内部的空间特征
        out = self.depthwise(x)

        # 再通过 1×1 卷积混合不同通道之间的信息
        out = self.pointwise(out)

        # 对特征进行归一化，缓解特征分布不稳定问题
        out = self.norm(out)

        # 通过激活函数引入非线性
        out = self.act(out)

        # 返回处理后的特征
        return out


# 残差缩放模块：用一个可学习系数控制增强特征加入原始特征的强度
class ResidualScale(nn.Module):

    def __init__(self, init_value=0.1):
        # 调用父类初始化函数
        super().__init__()

        # 定义可学习缩放系数，初始值较小，避免增强分支一开始过强
        self.scale = nn.Parameter(torch.tensor(float(init_value)))

    def forward(self, x, residual):
        # 将增强特征 residual 按可学习比例加入原始输入 x
        out = x + self.scale * residual

        # 返回残差增强后的特征
        return out


# 模块名称：001_频率结构协同弱纹理增强_工业裂纹弱边缘模块
# 模块作用：融合低频结构、高频细节、边缘梯度和背景上下文，增强工业裂纹等弱纹理缺陷特征
class FrequencyStructureDefectEnhancer001(nn.Module):

    def __init__(self, dim, reduction=4):
        # 调用父类初始化函数
        super().__init__()

        # 根据输入通道数计算动态权重生成分支中的隐藏通道数
        hidden = max(dim // reduction, 8)

        # 低频结构分支：先用平均池化平滑特征，再提取目标整体结构和轮廓信息
        self.low_structure = nn.Sequential(
            # 5×5 平均池化用于获得平滑后的低频信息
            nn.AvgPool2d(kernel_size=5, stride=1, padding=2),

            # 使用深度可分离卷积进一步提取低频结构特征
            DepthwiseSeparableConv(dim, kernel_size=5)
        )

        # 高频细节分支：用于提取局部纹理、细小裂纹和弱边缘响应
        self.high_detail = DepthwiseSeparableConv(
            dim,
            kernel_size=3
        )

        # 边缘细化分支：用于对 Sobel 梯度得到的边缘提示进行进一步增强
        self.edge_refine = DepthwiseSeparableConv(
            dim,
            kernel_size=3
        )

        # 背景抑制分支：提取大范围平滑背景信息，后续用于削弱无关背景响应
        self.background_suppress = nn.Sequential(
            # 9×9 平均池化用于获得更大感受野下的背景上下文
            nn.AvgPool2d(kernel_size=9, stride=1, padding=4),

            # 对背景上下文进行卷积建模
            DepthwiseSeparableConv(dim, kernel_size=5)
        )

        # 分支选择器：根据四个分支的拼接特征，自适应生成每个分支的权重
        self.selector = nn.Sequential(
            # 先将 4 倍通道数压缩到隐藏通道数，降低计算量
            ConvGNAct(dim * 4, hidden, kernel_size=1),

            # 再映射回 4 组权重，每组对应一个分支
            nn.Conv2d(hidden, dim * 4, kernel_size=1)
        )

        # 输出投影层：对融合后的增强特征进行通道整理
        self.proj = ConvGNAct(
            dim,
            dim,
            kernel_size=1,
            act=False
        )

        # 残差缩放层：控制增强特征注入原始特征的比例
        self.residual = ResidualScale(init_value=0.1)

        # 定义水平方向 Sobel 卷积核，用于检测左右方向的灰度变化
        sobel_x = torch.tensor(
            [
                [-1.0, 0.0, 1.0],
                [-2.0, 0.0, 2.0],
                [-1.0, 0.0, 1.0]
            ]
        ).view(1, 1, 3, 3)

        # 定义垂直方向 Sobel 卷积核，用于检测上下方向的灰度变化
        sobel_y = torch.tensor(
            [
                [-1.0, -2.0, -1.0],
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 1.0]
            ]
        ).view(1, 1, 3, 3)

        # 将水平方向 Sobel 卷积核复制到每个通道，并注册为 buffer，使其跟随模型移动到 GPU 但不参与训练
        self.register_buffer(
            'sobel_x',
            sobel_x.repeat(dim, 1, 1, 1)
        )

        # 将垂直方向 Sobel 卷积核复制到每个通道，并注册为 buffer，使其跟随模型移动到 GPU 但不参与训练
        self.register_buffer(
            'sobel_y',
            sobel_y.repeat(dim, 1, 1, 1)
        )

    def forward(self, x):
        # 获取输入特征的批量大小、通道数、高度和宽度
        b, c, h, w = x.shape

        # 通过低频结构分支提取平滑结构特征，突出目标整体轮廓
        low_frequency_feature = self.low_structure(x)

        # 用原始特征减去低频特征，得到更偏向高频细节的残差信息
        high_frequency_input = x - low_frequency_feature
        # 通过高频细节分支增强裂纹、边缘、细小纹理等局部响应
        high_frequency_feature = self.high_detail(high_frequency_input)

        # 使用水平方向 Sobel 卷积计算横向边缘梯度
        edge_x = F.conv2d(
            x,
            self.sobel_x,
            padding=1,
            groups=c
        )
        # 使用垂直方向 Sobel 卷积计算纵向边缘梯度
        edge_y = F.conv2d(
            x,
            self.sobel_y,
            padding=1,
            groups=c
        )
        # 将横向和纵向梯度绝对值相加，得到整体边缘强度图
        edge_strength = edge_x.abs() + edge_y.abs()
        # 使用 tanh 限制边缘响应范围，避免梯度幅值过大
        edge_strength = torch.tanh(edge_strength)
        # 通过边缘细化分支进一步提取稳定的边缘引导特征
        edge_guidance = self.edge_refine(edge_strength)

        # 通过背景分支提取大范围平滑背景上下文
        background_context = self.background_suppress(x)

        # 将低频结构、高频细节、边缘引导和背景上下文四类特征在通道维度拼接
        selector_input = torch.cat(
            [
                low_frequency_feature,
                high_frequency_feature,
                edge_guidance,
                background_context
            ],
            dim=1
        )
        # 通过分支选择器生成 4 组分支权重，输出形状为 [B, 4C, H, W]
        branch_weight = self.selector(selector_input)
        # 将权重 reshape 为 [B, 4, C, H, W]，其中 4 表示四个分支
        branch_weight = branch_weight.view(b, 4, c, h, w)
        # 在分支维度做 softmax，使四个分支的权重形成竞争关系
        branch_weight = torch.softmax(branch_weight, dim=1)

        # 根据动态权重融合低频结构特征
        low_enhanced = branch_weight[:, 0] * low_frequency_feature
        # 根据动态权重融合高频细节特征
        high_enhanced = branch_weight[:, 1] * high_frequency_feature
        # 根据动态权重融合边缘引导特征
        edge_enhanced = branch_weight[:, 2] * edge_guidance
        # 根据动态权重得到需要抑制的背景响应
        background_suppressed = branch_weight[:, 3] * background_context

        # 将结构、细节和边缘信息相加，同时减去背景上下文，实现缺陷增强和背景抑制
        enhanced_feature = (
            low_enhanced
            + high_enhanced
            + edge_enhanced
            - background_suppressed
        )

        # 对融合后的增强特征进行 1×1 投影，整理通道表达
        enhanced_feature = self.proj(enhanced_feature)
        # 将增强特征以可学习残差缩放的方式加回原始输入
        out = self.residual(x, enhanced_feature)
        # 返回最终输出特征，尺寸与输入保持一致
        return out

if __name__ == '__main__':
    input = torch.rand(1, 32, 256, 256)
    model = FrequencyStructureDefectEnhancer001(dim=32)
    output = model(input)
    print('input_size:', input.size())
    print('output_size:', output.size())
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：微创新·代码无误")
