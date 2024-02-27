import time

from mmdet.models import DETECTORS
# from mmdet.models import builder
from mmdet3d.models import builder
import torch
from projects.mmdet3d_plugin.models.detectors import BEVerse


@DETECTORS.register_module()
class FBverse(BEVerse):
    def __init__(self, pts_voxel_layer=None, pts_voxel_encoder=None, pts_middle_encoder=None, pts_fusion_layer=None,
                 img_backbone=None, pts_backbone=None, img_neck=None, pts_neck=None, transformer=None,
                 temporal_model=None, pts_bbox_head=None, img_roi_head=None, img_rpn_head=None, data_aug_conf=None,
                 train_cfg=None, test_cfg=None, pretrained=None, init_cfg=None, backward_projection=None,
                 depth_net=None):
        super().__init__(pts_voxel_layer, pts_voxel_encoder, pts_middle_encoder, pts_fusion_layer, img_backbone,
                         pts_backbone, img_neck, pts_neck, transformer, temporal_model, pts_bbox_head, img_roi_head,
                         img_rpn_head, data_aug_conf, train_cfg, test_cfg, pretrained, init_cfg)

        # BEVFormer init
        self.backward_projection = builder.build_head(backward_projection) if backward_projection else None
        # Depth Net
        self.depth_net = builder.build_head(depth_net) if depth_net else None

    def with_specific_component(self, component_name):
        """Whether the model owns a specific component"""
        return getattr(self, component_name, None) is not None

    def extract_img_feat(self, img, img_metas, future_egomotion=None,
                         aug_transform=None, img_is_valid=None, count_time=False, bev_mask=None):
        # image-view feature extraction
        imgs = img[0]

        B, S, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * S * N, C, imH, imW)
        x = self.img_backbone(imgs)

        if self.with_img_neck:
            x = self.img_neck(x)

        if isinstance(x, tuple):
            x_list = []
            for x_tmp in x:
                _, output_dim, ouput_H, output_W = x_tmp.shape
                x_list.append(x_tmp.view(B, N, output_dim, ouput_H, output_W))
            x = x_list
        else:
            _, output_dim, ouput_H, output_W = x.shape
            x = x.view(B, S, N, output_dim, ouput_H, output_W)

        cam_param = [i.flatten(0, 1) for i in img[1:]]
        if self.with_specific_component('depth_net'):
            mlp_input = self.depth_net.get_mlp_input(*cam_param)
            x, depth = self.depth_net(x.flatten(0, 1), mlp_input)
        else:
            x = x.flatten(0, 1)
            depth = None
        # lifting with LSS
        bev_feat = self.transformer([x.view(B, S, N, -1, ouput_H, output_W)] + img[1:6])
        if self.with_specific_component('backward_projection'):
            x = self.backward_projection([x],
                                         None,
                                         lss_bev=bev_feat.flatten(0, 1),
                                         cam_params=cam_param,
                                         bev_mask=bev_mask,
                                         gt_bboxes_3d=None,  # debug
                                         pred_img_depth=depth)
            bs, seq, chl, h, w = bev_feat.shape
            bev_feat = x.view(bs, seq, chl, h, w) + bev_feat

        torch.cuda.synchronize()
        t_BEV = time.time()

        # temporal processing
        bev_feat = self.temporal_model(bev_feat, future_egomotion=future_egomotion,
                                       aug_transform=aug_transform, img_is_valid=img_is_valid)

        torch.cuda.synchronize()
        t_temporal = time.time()

        if count_time:
            return bev_feat, {'t_BEV': t_BEV, 't_temporal': t_temporal}
        else:
            return bev_feat
