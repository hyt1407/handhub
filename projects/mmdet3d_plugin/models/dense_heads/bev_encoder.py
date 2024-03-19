from typing import Union, List, Tuple, Dict, Optional

import torch
import torch.nn as nn
from mmcv.runner import BaseModule, ModuleList
from mmdet.models.backbones.resnet import Bottleneck, BasicBlock
from mmcv.cnn import build_norm_layer, ConvModule, Conv2d, caffe2_xavier_init
from mmdet.models.utils import SinePositionalEncoding
from mmdet3d.models import builder
import torch.nn.functional as F
from torch import Tensor

from projects.mmdet3d_plugin.models.dense_heads.transformer import DetrTransformerDecoder


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, norm_cfg=dict(type='BN')):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        assert norm_cfg['type'] in ['BN', 'SyncBN']
        if norm_cfg['type'] == 'BN':
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=3, padding=1, bias=False),
                build_norm_layer(norm_cfg, out_channels, postfix=0)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=3, padding=1, bias=False),
                build_norm_layer(norm_cfg, out_channels, postfix=0)[1],
                nn.ReLU(inplace=True)
            )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class BevEncode(nn.Module):
    def __init__(self, numC_input, numC_output, num_layer=[2, 2, 2], num_channels=None,
                 backbone_output_ids=None, norm_cfg=dict(type='BN'), out_with_activision=False,
                 bev_encode_block='BottleNeck', multiview_learning=False, feature_fuse_type='SUM',
                 bev_encoder_fpn_type='lssfpn'):
        super(BevEncode, self).__init__()

        # build downsample modules for multiview learning
        self.multiview_learning = multiview_learning
        if self.multiview_learning:
            downsample_conv_list = []
            for i in range(len(num_layer) - 1):
                downsample_conv_list.append(
                    nn.Sequential(
                        nn.Conv2d(numC_input, numC_input * 2 ** (i + 1),
                                  kernel_size=3, stride=2 ** (i + 1), padding=1, bias=False),
                        build_norm_layer(norm_cfg, numC_input *
                                         2 ** (i + 1), postfix=0)[1],
                        nn.ReLU(inplace=True)))
            self.downsample_conv_list = nn.Sequential(*downsample_conv_list)
        self.feature_fuse_type = feature_fuse_type

        # build backbone
        assert len(num_layer) >= 3
        num_channels = [numC_input * 2 ** (i + 1) for i in range(
            len(num_layer))] if num_channels is None else num_channels

        # default: [128, 256, 512]

        # 输出最后三层特征
        self.backbone_output_ids = range(len(
            num_layer) - 3, len(num_layer)) if backbone_output_ids is None else backbone_output_ids

        layers = []
        if bev_encode_block == 'BottleNeck':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [Bottleneck(curr_numC, num_channels[i] // 4, stride=2,
                                    downsample=nn.Conv2d(
                                        curr_numC, num_channels[i], 3, 2, 1),
                                    norm_cfg=norm_cfg)]
                curr_numC = num_channels[i]
                layer.extend([Bottleneck(curr_numC, curr_numC // 4,
                                         norm_cfg=norm_cfg) for _ in range(num_layer[i] - 1)])
                layers.append(nn.Sequential(*layer))

        # [1/2, 1/4, 1/8]
        elif bev_encode_block == 'Basic':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [BasicBlock(curr_numC, num_channels[i], stride=2,
                                    downsample=nn.Conv2d(
                                        curr_numC, num_channels[i], 3, 2, 1),
                                    norm_cfg=norm_cfg)]
                curr_numC = num_channels[i]
                layer.extend([BasicBlock(curr_numC, curr_numC, norm_cfg=norm_cfg)
                              for _ in range(num_layer[i] - 1)])
                layers.append(nn.Sequential(*layer))
        else:
            assert False

        self.layers = nn.Sequential(*layers)

        # build neck
        self.bev_encoder_fpn_type = bev_encoder_fpn_type
        if self.bev_encoder_fpn_type == 'lssfpn':
            self.up1 = Up(num_channels[-1] + num_channels[-3],
                          numC_output * 2, scale_factor=4, norm_cfg=norm_cfg)
        elif self.bev_encoder_fpn_type == 'fpnv1':
            img_neck_cfg = dict(
                type='FPNv1',
                in_channels=num_channels[-3:],
                out_channels=numC_output * 2,
                num_outs=1,
                start_level=0,
                out_ids=[0])
            self.up1 = builder.build_neck(img_neck_cfg)
        elif self.bev_encoder_fpn_type == 'maskup':
            self.up1 = MaskUp(in_channels=[80, 160, 320, 640],
                              feat_channels=256,
                              out_channels=256,
                              num_queries=256,
                              enforce_decoder_input_project=False,
                              positional_encoding=dict(num_feats=128, normalize=True),  # SinePositionalEncoding
                              transformer_decoder=dict(return_intermediate=True,  # DetrTransformerDecoder
                                                       num_layers=6,
                                                       layer_cfg=dict(  # DetrTransformerDecoderLayer
                                                           self_attn_cfg=dict(  # MultiheadAttention
                                                               embed_dims=256,
                                                               num_heads=8,
                                                               attn_drop=0.1,
                                                               proj_drop=0.1,
                                                               dropout_layer=None,
                                                               batch_first=True),
                                                           cross_attn_cfg=dict(  # MultiheadAttention
                                                               embed_dims=256,
                                                               num_heads=8,
                                                               attn_drop=0.1,
                                                               proj_drop=0.1,
                                                               dropout_layer=None,
                                                               batch_first=True),
                                                           ffn_cfg=dict(
                                                               embed_dims=256,
                                                               feedforward_channels=2048,
                                                               num_fcs=2,
                                                               act_cfg=dict(type='ReLU', inplace=True),
                                                               ffn_drop=0.1,
                                                               dropout_layer=None,
                                                               add_identity=True)),
                                                       init_cfg=None), )
        else:
            assert False
        if self.bev_encoder_fpn_type != 'maskup':
            assert norm_cfg['type'] in ['BN', 'SyncBN']
            if norm_cfg['type'] == 'BN':
                self.up2 = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear',
                                align_corners=True),
                    nn.Conv2d(numC_output * 2, numC_output,
                              kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(numC_output),
                    nn.ReLU(inplace=True),
                )
            else:
                # 移除掉输出层的 linear conv, 使得输出为激活后的特征值
                self.up2 = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear',
                                align_corners=True),
                    nn.Conv2d(numC_output * 2, numC_output,
                              kernel_size=3, padding=1, bias=False),
                    build_norm_layer(norm_cfg, numC_output, postfix=0)[1],
                    nn.ReLU(inplace=True),
                )

            self.out_with_activision = out_with_activision
            if not self.out_with_activision:
                self.up2.add_module('4', nn.Conv2d(
                    numC_output, numC_output, kernel_size=1, padding=0))
                # self.up2.add_module('linear_output', nn.Conv2d(
                #     numC_output, numC_output, kernel_size=1, padding=0))

        self.fp16_enabled = False

    def forward(self, bev_feat_list):
        feats = []
        x_tmp = bev_feat_list[0]
        for lid, layer in enumerate(self.layers):
            x_tmp = layer(x_tmp)
            # x_tmp = checkpoint.checkpoint(layer,x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
            if lid < (len(self.layers) - 1) and self.multiview_learning:
                if self.feature_fuse_type == 'SUM':
                    bev_feat_from_img_view = bev_feat_list[lid + 1]
                    bev_feat_from_img_view = self.downsample_conv_list[lid](
                        bev_feat_from_img_view)
                    x_tmp = x_tmp + bev_feat_from_img_view
                else:
                    assert False

        if self.bev_encoder_fpn_type == 'lssfpn':
            res = self.up1(feats[-1], feats[-3])
        elif self.bev_encoder_fpn_type == 'fpnv1':
            res = self.up1(feats)
        elif self.bev_encoder_fpn_type == 'maskup':
            bev_feat_list.extend(feats)
            return self.up1(bev_feat_list)
        else:
            assert False

        res = self.up2(res)

        return res


class PixelDecoder(BaseModule):
    """Pixel decoder with a structure like fpn.

    Args:
        in_channels (list[int] | tuple[int]): Number of channels in the
            input feature maps.
        feat_channels (int): Number channels for feature.
        out_channels (int): Number channels for output.
        norm_cfg (:obj:`ConfigDict` or dict): Config for normalization.
            Defaults to dict(type='GN', num_groups=32).
        act_cfg (:obj:`ConfigDict` or dict): Config for activation.
            Defaults to dict(type='ReLU').
        encoder (:obj:`ConfigDict` or dict): Config for transorformer
            encoder.Defaults to None.
        positional_encoding (:obj:`ConfigDict` or dict): Config for
            transformer encoder position encoding. Defaults to
            dict(type='SinePositionalEncoding', num_feats=128,
            normalize=True).
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 in_channels: Union[List[int], Tuple[int]],
                 feat_channels: int,
                 out_channels: int,
                 norm_cfg: Dict = dict(type='GN', num_groups=32),
                 act_cfg: Dict = dict(type='ReLU'),
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.num_inputs = len(in_channels)
        self.lateral_convs = ModuleList()
        self.output_convs = ModuleList()
        self.use_bias = norm_cfg is None
        for i in range(0, self.num_inputs - 1):
            lateral_conv = ConvModule(
                in_channels[i],
                feat_channels,
                kernel_size=1,
                bias=self.use_bias,
                norm_cfg=norm_cfg)
            output_conv = ConvModule(
                feat_channels,
                feat_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=self.use_bias,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)

        self.last_feat_conv = ConvModule(
            in_channels[-1],
            feat_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=self.use_bias,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.mask_feature = Conv2d(
            feat_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def init_weights(self) -> None:
        """Initialize weights."""
        for i in range(0, self.num_inputs - 2):
            caffe2_xavier_init(self.lateral_convs[i].conv, bias=0)
            caffe2_xavier_init(self.output_convs[i].conv, bias=0)

        caffe2_xavier_init(self.mask_feature, bias=0)
        caffe2_xavier_init(self.last_feat_conv, bias=0)

    def forward(self, feats: List[Tensor],
                batch_img_metas: List[dict]) -> Tuple[Tensor, Tensor]:
        """
        Args:
            feats (list[Tensor]): Feature maps of each level. Each has
                shape of (batch_size, c, h, w).
            batch_img_metas (list[dict]): List of image information.
                Pass in for creating more accurate padding mask. Not
                used here.

        Returns:
            tuple[Tensor, Tensor]: a tuple containing the following:

                - mask_feature (Tensor): Shape (batch_size, c, h, w).
                - memory (Tensor): Output of last stage of backbone.\
                        Shape (batch_size, c, h, w).
        """
        y = self.last_feat_conv(feats[-1])
        for i in range(self.num_inputs - 2, -1, -1):
            x = feats[i]
            cur_feat = self.lateral_convs[i](x)
            y = cur_feat + \
                F.interpolate(y, size=cur_feat.shape[-2:], mode='nearest')
            y = self.output_convs[i](y)

        mask_feature = self.mask_feature(y)
        memory = feats[-1]
        return mask_feature, memory


class MaskUp(nn.Module):
    def __init__(self, in_channels: List[int],
                 out_channels: int,
                 feat_channels: int,
                 num_queries: int = 100,
                 enforce_decoder_input_project: bool = False,
                 positional_encoding: Dict = dict(num_feats=128, normalize=True),
                 transformer_decoder: Dict = ..., ):
        super().__init__()
        self.pixel_decoder = PixelDecoder(in_channels=in_channels,
                                          feat_channels=feat_channels,
                                          out_channels=out_channels,
                                          norm_cfg=dict(type='GN', num_groups=32),
                                          act_cfg=dict(type='ReLU'))
        self.decoder_pe = SinePositionalEncoding(**positional_encoding)
        self.transformer_decoder = DetrTransformerDecoder(
            **transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims
        if type(self.pixel_decoder) == PixelDecoder and (
                self.decoder_embed_dims != in_channels[-1]
                or enforce_decoder_input_project):
            self.decoder_input_proj = Conv2d(
                in_channels[-1], self.decoder_embed_dims, kernel_size=1)
        else:
            self.decoder_input_proj = nn.Identity()

        self.query_embed = nn.Embedding(num_queries, out_channels)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))

    def forward(self, x):
        batch_size = x[0].shape[0]
        padding_mask = x[-1].new_ones((batch_size, 200, 400),
                                      dtype=torch.float32)
        for i in range(batch_size):
            img_h, img_w = 200, 400
            padding_mask[i, :img_h, :img_w] = 0
        padding_mask = F.interpolate(
            padding_mask.unsqueeze(1), size=x[-1].shape[-2:],
            mode='nearest').to(torch.bool).squeeze(1)
        # when backbone is swin, memory is output of last stage of swin.
        # when backbone is r50, memory is output of tranformer encoder.
        mask_features, memory = self.pixel_decoder(x, None)
        pos_embed = self.decoder_pe(padding_mask)
        memory = self.decoder_input_proj(memory)
        # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
        memory = memory.flatten(2).permute(0, 2, 1)
        pos_embed = pos_embed.flatten(2).permute(0, 2, 1)
        # shape (batch_size, h * w)
        padding_mask = padding_mask.flatten(1)
        # shape = (num_queries, embed_dims)
        query_embed = self.query_embed.weight
        # shape = (batch_size, num_queries, embed_dims)
        query_embed = query_embed.unsqueeze(0).repeat(batch_size, 1, 1)
        target = torch.zeros_like(query_embed)
        # shape (num_decoder, num_queries, batch_size, embed_dims)
        out_dec = self.transformer_decoder(
            query=target,
            key=memory,
            value=memory,
            query_pos=query_embed,
            key_pos=pos_embed,
            key_padding_mask=padding_mask)

        # cls_scores
        mask_embed = self.mask_embed(out_dec)
        all_mask_preds = torch.einsum('lbqc,bchw->lbqhw', mask_embed,
                                      mask_features)

        return all_mask_preds[-1]
