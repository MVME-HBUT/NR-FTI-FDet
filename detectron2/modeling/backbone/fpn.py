# Copyright (c) Facebook, Inc. and its affiliates.
import math
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
from typing import List, Optional
#import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from detectron2.layers import (
    CNNBlockBase,
    Conv2d,
    DeformConv,
    ModulatedDeformConv,
    ShapeSpec,
    get_norm,
)

from detectron2.layers import Conv2d, ShapeSpec, get_norm

from .build import BACKBONE_REGISTRY
from .backbone import Backbone
#from resnet import build_resnet_backbone
#from mobilenetV3large import build_mobilenetV3large_backbone
#from CSPdarknet53 import build_CSPdarknet53_backbone
from .mobilenetV3small import build_mobilenetV3small_backbone

__all__ = ["build_resnet_fpn_backbone", "build_retinanet_resnet_fpn_backbone", "build_mobilenetV3large_fpn_backbone","build_mobilenetV3small_fpn_backbone","FPN","build_CSPdarknet53_FPN_backbone"]


def Have_a_Look(image,str):
    print(str)
    # [N, C, H, W] -> [C, H, W]
    im = np.squeeze(image.detach().cpu().numpy())
    # [C, H, W] -> [H, W, C]
    im = np.transpose(im, [1, 2, 0])
    im_mean = np.mean(im,axis=(2))

    # 查看这一层不同通道的图像，在这里有256层
    plt.figure()
    # for i in range(16):
    #     ax = plt.subplot(4, 4, i+1)
    #     plt.suptitle(str)
    #     plt.imshow(im[:, :, i], cmap='gray')
    plt.show()




class BottleneckBlock(CNNBlockBase):
    """
    The standard bottleneck residual block used by ResNet-50, 101 and 152
    defined in :paper:`ResNet`.  It contains 3 conv layers with kernels
    1x1, 3x3, 1x1, and a projection shortcut if needed.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels=16,
        stride=1,
        num_groups=1,
        norm="BN",
        stride_in_1x1=False,
        dilation=1,
    ):
        """
        Args:
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            num_groups (int): number of groups for the 3x3 conv layer.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            stride_in_1x1 (bool): when stride>1, whether to put stride in the
                first 1x1 convolution or the bottleneck 3x3 convolution.
            dilation (int): the dilation rate of the 3x3 conv layer.
        """
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        #self.se = SELayer1(out_channels)

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

        '''
        Arge:
            # Zero-initialize the last normalization in each residual branch,
            # so that at the beginning, the residual branch starts with zeros,
            # and each residual block behaves like an identity.
            # See Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
            # "For BN layers, the learnable scaling coefficient γ is initialized
            # to be 1, except for each residual block's last BN
            # where γ is initialized to be 0."

            # nn.init.constant_(self.conv3.norm.weight, 0)
            # TODO this somehow hurts performance when training GN models from scratch.
            # Add it as an option when we need to use this code to train a backbone.
        '''

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        out = self.conv2(out)
        out = F.relu_(out)

        out = self.conv3(out)

        # if y is not None:
        #     inrelu = self.se(y) #注意力快捷方式
        # else:
        #     inrelu = self.shortcut(x)
        
        # # if self.shortcut is not None:
        # #     shortcut = self.shortcut(x)
        # #     #shortcut = self.se(shortcut) #捷径添加注意力
        # # else:
        # #     shortcut = x
        # #     #shortcut = self.se(x) #捷径添加注意力

        # # out += shortcut
        # # out = F.relu_(out)  #将最后两步留到外面
        return out




class SELayer(nn.Module):
    def __init__(self, in_channels,channel,bias,norm, reduction=16):
        super(SELayer, self).__init__()
        self.conv1 = Conv2d(in_channels, channel, kernel_size=1, bias=bias, norm=norm)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)





class ASPP(nn.Module):
    def __init__(self, num_classes,in_channels):
        super(ASPP, self).__init__()

        self.conv_1x1_1 = nn.Conv2d(in_channels, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(256)

        self.conv_3x3_1 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1, dilation=1) #修改一下？1,2,5,第一次实验是6,12,18
        self.bn_conv_3x3_1 = nn.BatchNorm2d(256)

        self.conv_3x3_2 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=2, dilation=2)#保持原图大小不变 padding=dilation
        self.bn_conv_3x3_2 = nn.BatchNorm2d(256)

        self.conv_3x3_3 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=5, dilation=5)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(256)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_1x1_2 = nn.Conv2d(in_channels, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(256)

        #self.conv_1x1_3 = nn.Conv2d(1280, 256, kernel_size=1) # (1280 = 5*256)
        #self.conv_1x1_3 = nn.Conv2d(768, 256, kernel_size=1) # (768 = 3*256)
        self.conv_1x1_3 = nn.Conv2d(1024, 256, kernel_size=1) # (1024 = 4*256)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(256)

        self.conv_1x1_4 = nn.Conv2d(256, num_classes, kernel_size=1)
        
    def forward(self, feature_map):
        #--------------------#
        #   连续空洞卷积
        #--------------------#
        out = self.conv_3x3_1(feature_map)
        out = self.conv_3x3_2(out)
        out = self.conv_3x3_3(out)
        return out



class FPN(Backbone):
    """
    This module implements :paper:`FPN`.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(self, bottom_up, in_features, out_channels, norm="", top_block=None, fuse_type="sum"):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            norm (str): the normalization to use.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra FPN levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            fuse_type (str): types for fusing the top down features and the lateral
                ones. It can be "sum" (default), which sums up element-wise; or "avg",
                which takes the element-wise mean of the two.
        """
        super(FPN, self).__init__()
        assert isinstance(bottom_up, Backbone)
        assert in_features, in_features

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        input_shapes = bottom_up.output_shape()
        strides = [input_shapes[f].stride for f in in_features]
        in_channels_per_feature = [input_shapes[f].channels for f in in_features]

        

        #h = [input_shapes[f].height for f in in_features]
        #w = [input_shapes[f].width for f in in_features]

        _assert_strides_are_log2_contiguous(strides)
        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        #in_My_channels = 0
        for idx, in_channels in enumerate(in_channels_per_feature):
            lateral_norm = get_norm(norm, out_channels)
            output_norm = get_norm(norm, out_channels)
            stage = int(math.log2(strides[idx]))
            #加入含1X1卷积的注意力模块
            lateral_conv = SELayer(in_channels, out_channels, bias=use_bias, norm=lateral_norm)
            self.add_module("fpn_lateral{}".format(stage), lateral_conv)
            lateral_convs.append(lateral_conv)
            output_conv = BottleneckBlock(out_channels,out_channels)
            self.add_module("fpn_output{}".format(stage), output_conv)
            output_convs.append(output_conv)
            

        self.lateral_convs = lateral_convs[::-1] #翻转上下层的卷积
        self.output_convs = output_convs[::-1] #翻转上下层的卷积
        self.top_block = top_block
        self.in_features = in_features
        self.bottom_up = bottom_up
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type
        self.rev_in_features = tuple(in_features[::-1])
        self.deeplab = ASPP(256,576)
        self.deeplab_all=[]
        self.deeplab1 = ASPP(256,48)
        self.deeplab2 = ASPP(256,24)
        self.deeplab3 = ASPP(256,256)
        self.deeplab_all.append(self.deeplab1)
        self.deeplab_all.append(self.deeplab2)
        
        

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        bottom_up_features = self.bottom_up(x)
        results = []
        prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])

        shout_features = self.deeplab3(prev_features)
        Top_last_features = self.output_convs[0](prev_features) + shout_features
        results.append(Top_last_features)
        
        for features, lateral_conv, output_conv in zip(
            self.rev_in_features[1:], self.lateral_convs[1:], self.output_convs[1:]
        ):
            features = bottom_up_features[features]
            top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
            lateral_features = lateral_conv.forward(features)
            prev_features = lateral_features + top_down_features
            last_features = output_conv.forward(prev_features)
            if self._fuse_type == "avg":
                prev_features /= 2
            shoutcut_features = top_down_features
            last_features += shoutcut_features
            results.insert(0,last_features)
      

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))

        assert len(self._out_features) == len(results)
        return dict(list(zip(self._out_features, results)))

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )


class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


class LastLevelP6(nn.Module):
    """
    This module is used in FCOS to generate extra layers
    """

    def __init__(self, in_channels, out_channels, in_features="res5"):
        super().__init__()
        self.num_levels = 1 #表示增加的额外 FPN 级别的数量
        self.in_feature = in_features
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        for module in [self.p6]:
            weight_init.c2_xavier_fill(module)

    def forward(self, x):
        p6 = self.p6(x)
        return [p6]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7 from
    C5 feature.
    """

    def __init__(self, in_channels, out_channels, in_feature="res5"):
        super().__init__()
        self.num_levels = 2 #表示增加的额外 FPN 级别的数量
        self.in_feature = in_feature
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            weight_init.c2_xavier_fill(module)

    def forward(self, c5):
        p6 = self.p6(c5)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]



@BACKBONE_REGISTRY.register()
def build_mobilenetV3small_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_mobilenetV3small_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS

    #in_channels_p6p7 = bottom_up.output_shape()["res5"].channels

    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        #-------直接吧top_block改成None------##
        #top_block=LastLevelP6P7(in_channels_p6p7, out_channels),
        top_block=None,
        ##------------------------------##
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone

