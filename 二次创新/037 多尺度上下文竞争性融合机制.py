import torch
import torch.nn as nn
import torch.nn.functional as F

""" 
   多尺度上下文竞争性融合机制：
        写作思路与代码讲解：https://www.bilibili.com/video/BV18rVC63EfA/
        作用位置：任何单一特征处理时/任何普通卷积，或者任何即插即用模块中。
        主要功能（写作要点）：①通过多分支提取关键线索，缓解难以区分有效结构与背景响应的问题；
                        ②补充小尺度目标细节与大范围场景关系，缓解单一感受野难以适应尺度变化的问题。（将在本视频的写作部分展开阐述）
        代码层面：首先进入局部细节、半局部感受野和全局上下文分支，不同路径分别描述局部细节、结构上下文和显著区域；随后模块依据当前样本生成融合权重，将增强信息重新作用到原始特征上。
"""

# 基础卷积模块：Conv2d + GroupNorm + SiLU，用于构建轻量特征变换单元
class ConvGNAct(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, groups=1, act=True, dilation=1):
        # 初始化父类 nn.Module
        super().__init__()

        # 根据卷积核大小和空洞率计算 padding，使输出特征图尺寸尽量保持不变
        padding = dilation * (kernel_size // 2)

        # 构建卷积、归一化和激活组成的基础模块
        self.block = nn.Sequential(
            # 使用二维卷积完成空间特征提取或通道数变换
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False
            ),

            # 使用 GroupNorm 稳定特征分布，这里 groups=1 类似按通道整体归一化
            nn.GroupNorm(1, out_channels),

            # 根据 act 参数决定是否使用 SiLU 激活函数
            nn.SiLU(inplace=True) if act else nn.Identity()
        )

    def forward(self, x):
        # 将输入特征送入基础卷积模块
        out = self.block(x)

        # 返回卷积、归一化和激活后的特征
        return out

# 深度可分离卷积模块：用较低计算量提取空间特征并融合通道信息
class DepthwiseSeparableConv(nn.Module):

    def __init__(self, dim, kernel_size=3, dilation=1):
        # 初始化父类 nn.Module
        super().__init__()

        # 根据卷积核大小和空洞率计算 padding，保持空间尺寸不变
        padding = dilation * (kernel_size // 2)

        # 深度卷积：每个通道单独做卷积，用于提取通道内的空间纹理
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

        # 对卷积后的特征进行归一化，提升训练稳定性
        self.norm = nn.GroupNorm(1, dim)

        # 使用 SiLU 激活函数增强非线性表达能力
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        # 先用深度卷积提取每个通道内部的空间上下文
        out = self.depthwise(x)

        # 再用 1×1 卷积融合不同通道的信息
        out = self.pointwise(out)

        # 对融合后的特征进行归一化
        out = self.norm(out)

        # 对归一化后的特征进行非线性激活
        out = self.act(out)

        # 返回深度可分离卷积处理后的特征
        return out

# 残差缩放模块：用可学习系数控制增强分支对原始特征的影响强度
class ResidualScale(nn.Module):

    def __init__(self, init_value=0.1):
        # 初始化父类 nn.Module
        super().__init__()

        # 定义一个可学习的缩放参数，初始值较小，避免增强分支一开始破坏原始特征
        self.scale = nn.Parameter(torch.tensor(float(init_value)))

    def forward(self, x, residual):
        # 将增强特征 residual 按可学习比例加入原始输入 x
        out = x + self.scale * residual

        # 返回残差增强后的特征
        return out


# 模块名称：002_轻量长程上下文尺度竞争融合_密集小目标检测模块
# 模块作用：融合局部细节、中尺度上下文和长程全局上下文，增强密集小目标的多尺度表达能力
class LightweightLongRangeScaleFusion002(nn.Module):

    def __init__(self, dim, reduction=4, proxy_size=16):
        # 初始化父类 nn.Module
        super().__init__()

        # 根据输入通道数计算隐藏通道数，用于降低动态权重分支的计算量
        hidden = max(dim // reduction, 8)

        # 保存代理上下文特征的空间尺寸，例如 16 表示压缩到 16×16 的全局代理网格
        self.proxy_size = proxy_size

        # 局部分支：在原始分辨率下提取细粒度纹理和小目标局部特征
        self.local_branch = DepthwiseSeparableConv(
            dim,
            kernel_size=3
        )

        # 中尺度分支：使用更大的卷积核和空洞率扩大感受野，补充邻域上下文信息
        self.mid_branch = DepthwiseSeparableConv(
            dim,
            kernel_size=5,
            dilation=2
        )

        # 全局代理分支：在低分辨率代理特征上建模长程上下文，降低全局建模成本
        self.global_proxy = nn.Sequential(
            # 在代理特征图上提取上下文信息
            DepthwiseSeparableConv(dim, kernel_size=3),

            # 使用 1×1 卷积进一步整理通道特征
            ConvGNAct(dim, dim, kernel_size=1)
        )

        # 通道摘要分支：根据全局平均池化结果生成通道级注意力权重
        self.channel_summary = nn.Sequential(
            # 将每个通道压缩成一个全局统计值
            nn.AdaptiveAvgPool2d(1),

            # 将通道数压缩到 hidden，减少参数量
            ConvGNAct(dim, hidden, kernel_size=1),

            # 将 hidden 通道重新映射回原始通道数
            nn.Conv2d(hidden, dim, kernel_size=1),

            # 使用 Sigmoid 将通道权重限制在 0 到 1 之间
            nn.Sigmoid()
        )

        # 尺度路由器：根据局部、中尺度和全局代理特征，动态生成三个尺度分支的融合权重
        self.scale_router = nn.Sequential(
            # 将三个分支拼接后的 3C 通道压缩到 hidden 通道
            ConvGNAct(dim * 3, hidden, kernel_size=1),

            # 输出 3 个尺度权重，分别对应 local、mid 和 global proxy 分支
            nn.Conv2d(hidden, 3, kernel_size=1)
        )

        # 空间门控分支：判断当前位置更应该依赖多尺度上下文还是通道注意力特征
        self.spatial_gate = nn.Sequential(
            # 输入由 mixed_context 和通道调制特征拼接而成，因此通道数为 2C
            ConvGNAct(dim * 2, hidden, kernel_size=3),

            # 输出单通道空间门控图，对每个空间位置给出融合权重
            nn.Conv2d(hidden, 1, kernel_size=1),

            # 使用 Sigmoid 将空间门控值限制在 0 到 1 之间
            nn.Sigmoid()
        )

        # 输出投影层：对融合后的增强特征进行通道整理
        self.proj = ConvGNAct(
            dim,
            dim,
            kernel_size=1,
            act=False
        )

        # 残差缩放层：将增强特征稳定地加入原始输入
        self.residual = ResidualScale(init_value=0.1)

    def forward(self, x):
        # 获取输入特征的批量大小、通道数、高度和宽度
        b, c, h, w = x.shape

        # 通过局部分支提取原始分辨率下的细节纹理特征
        local_feature = self.local_branch(x)

        # 通过中尺度分支提取更大邻域范围内的上下文特征
        mid_context = self.mid_branch(x)

        # 将输入特征自适应池化到固定大小，得到低分辨率全局代理特征
        proxy_context = F.adaptive_avg_pool2d(
            x,
            self.proxy_size
        )
        # 在低分辨率代理特征上进行轻量上下文建模
        proxy_context = self.global_proxy(proxy_context)
        # 将全局代理上下文上采样回原始输入的空间尺寸
        proxy_context = F.interpolate(
            proxy_context,
            size=(h, w),
            mode='bilinear',
            align_corners=False
        )

        # 将局部特征、中尺度上下文和全局代理上下文在通道维度拼接
        router_input = torch.cat(
            [
                local_feature,
                mid_context,
                proxy_context
            ],
            dim=1
        )
        # 通过尺度路由器生成三个分支的动态权重
        scale_weight = self.scale_router(router_input)
        # 在尺度维度上进行 softmax，使三个分支之间形成竞争关系
        scale_weight = torch.softmax(
            scale_weight,
            dim=1
        )

        # 使用第 1 个权重加权局部分支特征
        local_weighted = scale_weight[:, 0:1] * local_feature
        # 使用第 2 个权重加权中尺度上下文特征
        mid_weighted = scale_weight[:, 1:2] * mid_context
        # 使用第 3 个权重加权全局代理上下文特征
        proxy_weighted = scale_weight[:, 2:3] * proxy_context
        # 将三个尺度分支的加权结果相加，得到多尺度上下文融合特征
        mixed_context = local_weighted + mid_weighted + proxy_weighted

        # 根据输入特征生成通道级注意力权重
        channel_gate = self.channel_summary(x)
        # 使用通道权重对原始输入进行调制，突出重要通道
        channel_refined = x * channel_gate

        # 将多尺度上下文特征和通道调制特征拼接，作为空间门控的输入
        spatial_gate_input = torch.cat(
            [
                mixed_context,
                channel_refined
            ],
            dim=1
        )
        # 生成空间门控图，决定每个位置更偏向多尺度上下文还是通道调制特征
        spatial_gate = self.spatial_gate(spatial_gate_input)
        # 空间门控值越大，该位置越依赖多尺度上下文特征
        context_part = mixed_context * spatial_gate

        # 空间门控值越小，该位置越保留通道调制后的原始特征
        channel_part = channel_refined * (1.0 - spatial_gate)
        # 融合上下文增强特征和通道增强特征
        enhanced_feature = context_part + channel_part
        # 使用 1×1 投影层整理融合后的特征表达
        enhanced_feature = self.proj(enhanced_feature)
        # 通过可学习残差缩放将增强特征加回原始输入
        out = self.residual(x, enhanced_feature)
        # 返回最终输出特征，形状与输入保持一致
        return out

if __name__ == '__main__':
    input = torch.rand(1, 32, 256, 256)
    model = LightweightLongRangeScaleFusion002(dim=32)
    output = model(input)
    print('input_size:', input.size())
    print('output_size:', output.size())
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：微创新·代码无误")