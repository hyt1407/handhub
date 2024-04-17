_base_ = ['./beverse_small.py']

model = dict(
    img_backbone=dict(
        _delete_=True,
        type='ConvNeXt',
        arch='pico',
        drop_path_rate=0.1,
        layer_scale_init_value=0.,
        use_grn=True,
        out_indices=(2, 3,),
        gap_before_final_norm=False
    ),
    img_neck=dict(
        # type='FPN_LSS', in_channels=320 + 640, inverse=True, )  # nano
    type='FPN_LSS', in_channels=256 + 512, inverse=True, )# pico
    # type='FPN_LSS', in_channels=160 + 320, inverse=True, ) # atto
)
