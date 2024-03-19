_base_ = ['./beverse_convnext_v2.py']
use_checkpoint = True
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
grid_config = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-10.0, 10.0, 20.0],
    'depth': [2.0, 42.0, 0.5],
}
depth_categories = 80  # (grid_config['depth'][1]-grid_config['depth'][0])//grid_config['depth'][2]
grid_config_bevformer = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-10.0, 10.0, 20.0],
}
_dim_ = 512
_pos_dim_ = 40
bev_h_ = 128
bev_w_ = 128
numC_Trans = depth_categories
_num_levels_ = 1
_ffn_dim_ = numC_Trans * 4
receptive_field = 3
model = dict(
    type='FBverse',
    transformer=dict(numC_input=80,
                     numC_Trans=80),
    pts_bbox_head=dict(in_channels=80,bev_encoder_fpn_type='maskup',),
    temporal_model=dict(in_channels=80, start_out_channels=80),
    depth_net=dict(type='CM_DepthNet',  # camera-aware depth net
                   in_channels=_dim_,
                   context_channels=numC_Trans,
                   downsample=16,
                   grid_config=grid_config,
                   depth_channels=depth_categories,
                   with_cp=use_checkpoint,
                   loss_depth_weight=1.,
                   use_dcn=False),
    backward_projection=dict(
        type='BackwardProjection',
        bev_h=bev_h_,
        bev_w=bev_w_,
        in_channels=numC_Trans,
        out_channels=numC_Trans,
        pc_range=point_cloud_range,
        transformer=dict(
            type='BEVFormer',
            use_cams_embeds=False,
            embed_dims=numC_Trans,
            encoder=dict(
                type='BevformerEncoder',
                num_layers=1,
                pc_range=point_cloud_range,
                grid_config=grid_config_bevformer,
                data_config={'input_size': (512, 1408)},
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerEncoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=numC_Trans,
                            dropout=0.0,
                            num_levels=1),
                        dict(
                            type='DA_SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            dbound=[2.0, 42.0, 0.5],
                            dropout=0.0,
                            deformable_attention=dict(
                                type='DA_MSDeformableAttention',
                                embed_dims=numC_Trans,
                                num_points=8,
                                num_levels=_num_levels_),
                            embed_dims=numC_Trans,
                        )
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=numC_Trans,
                        feedforward_channels=_ffn_dim_,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True), ),
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
        ),
        positional_encoding=dict(
            type='CustormLearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
    ),
)

data = dict(
    train=dict(dataset=dict(type='RobodriveDatasetPseudo')),
    val=dict(type='RobodriveDatasetPseudo'),
    test=dict(pseudo_bda=True)
)
