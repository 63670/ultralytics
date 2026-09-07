import torch
import torch.nn as nn
import torch.nn.functional as F

""" 
   基于峰值的背景解耦机制：
        写作思路与代码讲解：https://www.bilibili.com/video/BV12S7f6RE4c/
        作用位置：任何单一特征处理时/任何普通卷积，或者任何即插即用模块中。
        主要功能（写作要点）：①通过峰值显著性增强有效结构与背景响应的问题；
                            ②强化微小目标与弱缺陷的局部峰值响应，并削弱复杂背景纹理带来的干扰。（将在本视频的写作部分展开阐述）
        代码层面：首先进入峰值显著性、低频背景抑制和弱边界补偿分支，分别描述局部细节、结构上下文和显著区域；依据当前融合权重响应，将增强信息重新作用到原始特征上。
"""

# 基础卷积模块：Conv2d + GroupNorm + SiLU，用于构建通用特征变换单元
class ConvGNAct(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, groups=1, act=True, dilation=1):
        # 初始化父类 nn.Module
        super().__init__()

        # 根据卷积核大小和空洞率计算 padding，使输出特征图尺寸尽量保持不变
        padding = dilation * (kernel_size // 2)

        # 构建卷积、归一化和激活函数组成的基础特征处理模块
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

            # 使用 GroupNorm 稳定特征分布，适合小 batch 或工业图像检测任务
            nn.GroupNorm(1, out_channels),

            # 如果 act=True，则使用 SiLU 激活函数；否则使用恒等映射
            nn.SiLU(inplace=True) if act else nn.Identity()
        )

    def forward(self, x):
        # 将输入特征送入基础卷积模块
        out = self.block(x)

        # 返回处理后的特征
        return out


# 深度可分离卷积模块：先逐通道提取空间信息，再用 1×1 卷积融合通道信息
class DepthwiseSeparableConv(nn.Module):

    def __init__(self, dim, kernel_size=3, dilation=1):
        # 初始化父类 nn.Module
        super().__init__()

        # 根据卷积核大小和空洞率计算 padding，保持输入输出空间尺寸一致
        padding = dilation * (kernel_size // 2)

        # 深度卷积：每个通道单独卷积，用较低计算量提取空间纹理特征
        self.depthwise = nn.Conv2d(
            dim,
            dim,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=dim,
            bias=False
        )

        # 逐点卷积：使用 1×1 卷积混合不同通道之间的信息
        self.pointwise = nn.Conv2d(
            dim,
            dim,
            kernel_size=1,
            bias=False
        )

        # 对卷积后的特征进行归一化，提升训练稳定性
        self.norm = nn.GroupNorm(1, dim)

        # 使用 SiLU 激活函数增强非线性表达能力
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        # 先通过深度卷积提取每个通道内部的空间纹理
        out = self.depthwise(x)

        # 再通过 1×1 卷积融合不同通道之间的信息
        out = self.pointwise(out)

        # 对通道融合后的特征进行归一化
        out = self.norm(out)

        # 使用激活函数增强非线性表达
        out = self.act(out)

        # 返回深度可分离卷积处理后的特征
        return out


# 残差缩放模块：使用可学习系数控制增强分支加入原始特征的强度
class ResidualScale(nn.Module):

    def __init__(self, init_value=0.1):
        # 初始化父类 nn.Module
        super().__init__()

        # 定义可学习缩放参数，初始值较小，避免训练初期增强分支扰动过强
        self.scale = nn.Parameter(torch.tensor(float(init_value)))

    def forward(self, x, residual):
        # 将增强特征 residual 按可学习比例加入原始输入 x
        out = x + self.scale * residual

        # 返回残差增强后的特征
        return out


# 模块名称：198_小目标弱缺陷峰值背景解耦_高分辨率恢复模块
# 模块作用：突出小目标、弱缺陷和局部峰值响应，同时抑制大范围平滑背景干扰
class TinyDefectPeakBackgroundDecoupler198(nn.Module):

    def __init__(self, dim, reduction=4):
        # 初始化父类 nn.Module
        super().__init__()

        # 根据输入通道数计算隐藏通道数，用于降低门控分支的计算量
        hidden = max(dim // reduction, 8)

        # 峰值细化分支：用于增强局部显著响应，例如小目标亮点、弱缺陷峰值和异常纹理
        self.peak_refine = DepthwiseSeparableConv(
            dim,
            kernel_size=3
        )

        # 背景建模分支：通过大核平均池化获得平滑背景，再进行卷积细化
        self.background_refine = nn.Sequential(
            # 9×9 平均池化用于提取大范围平滑背景上下文
            nn.AvgPool2d(
                kernel_size=9,
                stride=1,
                padding=4
            ),

            # 对背景上下文进行进一步建模
            DepthwiseSeparableConv(
                dim,
                kernel_size=5
            )
        )

        # 弱边缘细化分支：用于增强小目标或弱缺陷周围的细微边界变化
        self.weak_edge_refine = DepthwiseSeparableConv(
            dim,
            kernel_size=3
        )

        # 全局上下文分支：将特征压缩到固定大小，提取更大范围的语义上下文
        self.context_refine = nn.Sequential(
            # 自适应平均池化到 16×16，用低分辨率代理特征表达全局上下文
            nn.AdaptiveAvgPool2d(16),

            # 在低分辨率上下文特征上进行轻量卷积建模
            DepthwiseSeparableConv(
                dim,
                kernel_size=3
            )
        )

        # 目标门控分支：根据峰值、弱边缘、上下文和原始特征生成目标增强权重
        self.target_gate = nn.Sequential(
            # 四类特征拼接后通道数为 4C，先压缩到 hidden 通道
            ConvGNAct(
                dim * 4,
                hidden,
                kernel_size=1
            ),

            # 将 hidden 通道映射回 C 通道，生成每个通道的目标响应权重
            nn.Conv2d(
                hidden,
                dim,
                kernel_size=1
            ),

            # 使用 Sigmoid 将目标门控值限制在 0 到 1 之间
            nn.Sigmoid()
        )

        # 背景抑制门控分支：根据背景特征和原始特征生成背景抑制权重
        self.suppress_gate = nn.Sequential(
            # 背景特征与原始特征拼接后通道数为 2C，先压缩到 hidden 通道
            ConvGNAct(
                dim * 2,
                hidden,
                kernel_size=1
            ),

            # 将 hidden 通道映射回 C 通道，生成每个通道的背景抑制权重
            nn.Conv2d(
                hidden,
                dim,
                kernel_size=1
            ),

            # 使用 Sigmoid 将背景抑制权重限制在 0 到 1 之间
            nn.Sigmoid()
        )

        # 输出投影层：对增强后的特征进行通道整理
        self.proj = ConvGNAct(
            dim,
            dim,
            kernel_size=1,
            act=False
        )

        # 残差缩放层：将增强特征稳定地注入原始输入
        self.residual = ResidualScale(init_value=0.1)

    def forward(self, x):
        # 获取输入特征的批量大小、通道数、高度和宽度
        b, c, h, w = x.shape

        # 使用 3×3 最大池化提取局部峰值响应
        local_max = F.max_pool2d(
            x,
            kernel_size=3,
            stride=1,
            padding=1
        )
        # 使用 3×3 平均池化提取局部平滑背景参照
        local_mean = F.avg_pool2d(
            x,
            kernel_size=3,
            stride=1,
            padding=1
        )
        # 用局部最大响应减去局部平均响应，得到突出异常峰值的小目标特征
        peak_proxy = local_max - local_mean
        # 对峰值代理特征进行卷积细化，增强小目标/弱缺陷峰值响应
        peak_feature = self.peak_refine(peak_proxy)

        # 使用 5×5 平均池化获得更大邻域内的平滑参照
        edge_mean = F.avg_pool2d(
            x,
            kernel_size=5,
            stride=1,
            padding=2
        )
        # 用原始特征减去邻域平均特征，突出弱边缘和局部细节变化
        weak_edge_proxy = x - edge_mean
        # 对弱边缘代理特征进行卷积细化，增强边界和纹理细节
        weak_edge_feature = self.weak_edge_refine(weak_edge_proxy)

        # 将输入特征压缩到固定空间尺寸，提取低分辨率全局上下文
        context_feature = self.context_refine(x)
        # 将全局上下文特征上采样回原始输入特征的空间尺寸
        context_feature = F.interpolate(
            context_feature,
            size=(h, w),
            mode='bilinear',
            align_corners=False
        )

        # 将峰值特征、弱边缘特征、上下文特征和原始特征拼接，作为目标门控输入
        target_gate_input = torch.cat(
            [
                peak_feature,
                weak_edge_feature,
                context_feature,
                x
            ],dim=1
        )
        # 生成目标增强门控，用于控制哪些区域和通道需要被强化
        target_gate = self.target_gate(target_gate_input)
        # 将峰值、弱边缘和上下文特征相加，形成目标相关增强信息
        target_feature = (peak_feature+ weak_edge_feature+ context_feature)
        # 使用目标门控强化目标相关特征
        target_enhanced = target_gate * target_feature

        # 通过背景建模分支提取大范围平滑背景特征
        background_feature = self.background_refine(x)
        # 将背景特征和原始特征拼接，作为背景抑制门控输入
        suppress_gate_input = torch.cat([background_feature, x],dim=1)
        # 生成背景抑制门控，用于控制哪些背景响应需要被削弱
        suppress_gate = self.suppress_gate(suppress_gate_input)
        # 使用背景抑制门控提取需要扣除的背景干扰
        background_suppressed = suppress_gate * background_feature
        # 目标增强特征减去背景干扰特征，实现小目标弱缺陷与背景的解耦
        enhanced_feature = target_enhanced - background_suppressed

        # 通过 1×1 投影层整理增强特征的通道表达
        enhanced_feature = self.proj(enhanced_feature)
        # 将增强特征以可学习残差缩放方式加回原始输入
        out = self.residual(x, enhanced_feature)
        # 返回最终输出特征，尺寸与输入保持一致
        return out

if __name__ == '__main__':
    # 构造一个随机输入特征，形状为 [B, C, H, W]
    input = torch.rand(1, 32, 256, 256)
    # 实例化小目标弱缺陷峰值背景解耦模块，输入通道数设置为 32
    model = TinyDefectPeakBackgroundDecoupler198(dim=32)
    # 将输入特征送入模型，执行一次前向传播
    output = model(input)
    # 打印输入特征尺寸，检查输入是否符合 BCHW 格式
    print('input_size:', input.size())
    # 打印输出特征尺寸，检查模块是否保持输入输出尺寸一致
    print('output_size:', output.size())
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：微创新·代码无误")