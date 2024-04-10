point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
dataset_type = 'RobodriveDataset'
data_root = 'data/nuscenes/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
    prototype='lift-splat-shoot')
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_MTL',
        using_ego=True,
        temporal_consist=True,
        is_train=True,
        data_aug_conf=dict(
            resize_lim=(0.82, 0.99),
            final_dim=(512, 1408),
            rot_lim=(-5.4, 5.4),
            H=900,
            W=1600,
            rand_flip=True,
            bot_pct_lim=(0.0, 0.22),
            crop_h=(0.0, 0.0),
            cams=[
                'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
            ],
            Ncams=6)),
    dict(
        type='LoadAnnotations3D_MTL',
        with_bbox_3d=True,
        with_label_3d=True,
        with_instance_tokens=True),
    dict(
        type='MTLGlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        update_img2lidar=True),
    dict(
        type='MTLRandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        update_img2lidar=True),
    dict(
        type='RasterizeMapVectors',
        map_grid_conf=dict(
            xbound=[-30.0, 30.0, 0.15],
            ybound=[-15.0, 15.0, 0.15],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[1.0, 60.0, 1.0])),
    dict(
        type='ConvertMotionLabels',
        grid_conf=dict(
            xbound=[-50.0, 50.0, 0.5],
            ybound=[-50.0, 50.0, 0.5],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[1.0, 60.0, 1.0]),
        only_vehicle=True),
    dict(type='ObjectValidFilter'),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='Collect3D',
        keys=[
            'img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'semantic_indices',
            'semantic_map', 'future_egomotions', 'aug_transform',
            'img_is_valid', 'motion_segmentation', 'motion_instance',
            'instance_centerness', 'instance_offset', 'instance_flow',
            'has_invalid_frame'
        ],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                   'depth2img', 'cam2img', 'pad_shape', 'lidar2ego_rots',
                   'lidar2ego_trans', 'scale_factor', 'flip',
                   'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
                   'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
                   'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                   'transformation_3d_flow', 'img_info'))
]
test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_MTL',
        using_ego=True,
        data_aug_conf=dict(
            resize_lim=(0.82, 0.99),
            final_dim=(512, 1408),
            rot_lim=(-5.4, 5.4),
            H=900,
            W=1600,
            rand_flip=True,
            bot_pct_lim=(0.0, 0.22),
            crop_h=(0.0, 0.0),
            cams=[
                'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
            ],
            Ncams=6)),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(
                type='Collect3D',
                keys=[
                    'img_inputs', 'future_egomotions', 'has_invalid_frame',
                    'img_is_valid'
                ],
                meta_keys=('filename', 'sample_idx', 'ori_shape', 'img_shape',
                           'lidar2img', 'depth2img', 'cam2img', 'pad_shape',
                           'scale_factor', 'flip', 'pcd_horizontal_flip',
                           'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                           'img_norm_cfg', 'pcd_trans', 'sample_idx',
                           'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                           'transformation_3d_flow', 'img_info',
                           'lidar2ego_rots', 'lidar2ego_trans'))
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=dict(backend='disk')),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=6,
    train=dict(
        type='CBGSDataset',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/nuscenes_infos_train.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[-0.3925, 0.3925],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='ObjectNameFilter',
                classes=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(type='PointShuffle'),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(
                type='Collect3D',
                keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ],
        classes=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        modality=dict(
            use_lidar=True,
            use_camera=False,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=False,
        box_type_3d='LiDAR',
        dataset=dict(
            type='RobodriveDatasetPseudo',
            data_root='data/nuscenes/',
            ann_file='data/nuscenes_infos/nuscenes_infos_train.pkl',
            pipeline=[
                dict(
                    type='LoadMultiViewImageFromFiles_MTL',
                    using_ego=True,
                    temporal_consist=True,
                    is_train=True,
                    data_aug_conf=dict(
                        resize_lim=(0.82, 0.99),
                        final_dim=(512, 1408),
                        rot_lim=(-5.4, 5.4),
                        H=900,
                        W=1600,
                        rand_flip=True,
                        bot_pct_lim=(0.0, 0.22),
                        crop_h=(0.0, 0.0),
                        cams=[
                            'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                        ],
                        Ncams=6)),
                dict(
                    type='LoadAnnotations3D_MTL',
                    with_bbox_3d=True,
                    with_label_3d=True,
                    with_instance_tokens=True),
                dict(
                    type='MTLGlobalRotScaleTrans',
                    rot_range=[-0.3925, 0.3925],
                    scale_ratio_range=[0.95, 1.05],
                    translation_std=[0, 0, 0],
                    update_img2lidar=True),
                dict(
                    type='MTLRandomFlip3D',
                    sync_2d=False,
                    flip_ratio_bev_horizontal=0.5,
                    flip_ratio_bev_vertical=0.5,
                    update_img2lidar=True),
                dict(
                    type='RasterizeMapVectors',
                    map_grid_conf=dict(
                        xbound=[-30.0, 30.0, 0.15],
                        ybound=[-15.0, 15.0, 0.15],
                        zbound=[-10.0, 10.0, 20.0],
                        dbound=[1.0, 60.0, 1.0])),
                dict(
                    type='ConvertMotionLabels',
                    grid_conf=dict(
                        xbound=[-50.0, 50.0, 0.5],
                        ybound=[-50.0, 50.0, 0.5],
                        zbound=[-10.0, 10.0, 20.0],
                        dbound=[1.0, 60.0, 1.0]),
                    only_vehicle=True),
                dict(type='ObjectValidFilter'),
                dict(
                    type='ObjectRangeFilter',
                    point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
                dict(
                    type='ObjectNameFilter',
                    classes=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='Collect3D',
                    keys=[
                        'img_inputs', 'gt_bboxes_3d', 'gt_labels_3d',
                        'semantic_indices', 'semantic_map',
                        'future_egomotions', 'aug_transform', 'img_is_valid',
                        'motion_segmentation', 'motion_instance',
                        'instance_centerness', 'instance_offset',
                        'instance_flow', 'has_invalid_frame'
                    ],
                    meta_keys=('filename', 'ori_shape', 'img_shape',
                               'lidar2img', 'depth2img', 'cam2img',
                               'pad_shape', 'lidar2ego_rots',
                               'lidar2ego_trans', 'scale_factor', 'flip',
                               'pcd_horizontal_flip', 'pcd_vertical_flip',
                               'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                               'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                               'pcd_rotation', 'pts_filename',
                               'transformation_3d_flow', 'img_info'))
            ],
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            test_mode=False,
            use_valid_flag=True,
            receptive_field=2,
            future_frames=3,
            grid_conf=dict(
                xbound=[-51.2, 51.2, 0.8],
                ybound=[-51.2, 51.2, 0.8],
                zbound=[-10.0, 10.0, 20.0],
                dbound=[1.0, 60.0, 1.0]),
            map_grid_conf=dict(
                xbound=[-30.0, 30.0, 0.15],
                ybound=[-15.0, 15.0, 0.15],
                zbound=[-10.0, 10.0, 20.0],
                dbound=[1.0, 60.0, 1.0]),
            modality=dict(
                use_camera=True,
                use_lidar=False,
                use_radar=False,
                use_map=False,
                use_external=False,
                prototype='lift-splat-shoot'),
            box_type_3d='LiDAR'),
        val=dict(type='RobodriveDatasetPseudo')),
    val=dict(
        type='MTLEgoNuScenesDataset',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes_infos/robodrive_infos_test.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles_MTL',
                using_ego=True,
                data_aug_conf=dict(
                    resize_lim=(0.82, 0.99),
                    final_dim=(512, 1408),
                    rot_lim=(-5.4, 5.4),
                    H=900,
                    W=1600,
                    rand_flip=True,
                    bot_pct_lim=(0.0, 0.22),
                    crop_h=(0.0, 0.0),
                    cams=[
                        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6)),
            dict(
                type='LoadAnnotations3D_MTL',
                with_bbox_3d=True,
                with_label_3d=True,
                with_instance_tokens=True),
            dict(
                type='RasterizeMapVectors',
                map_grid_conf=dict(
                    xbound=[-30.0, 30.0, 0.15],
                    ybound=[-15.0, 15.0, 0.15],
                    zbound=[-10.0, 10.0, 20.0],
                    dbound=[1.0, 60.0, 1.0])),
            dict(
                type='ConvertMotionLabels',
                grid_conf=dict(
                    xbound=[-50.0, 50.0, 0.5],
                    ybound=[-50.0, 50.0, 0.5],
                    zbound=[-10.0, 10.0, 20.0],
                    dbound=[1.0, 60.0, 1.0]),
                only_vehicle=True),
            dict(type='ObjectValidFilter'),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
            dict(
                type='ObjectNameFilter',
                classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ]),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=[
                            'img_inputs', 'semantic_indices', 'semantic_map',
                            'future_egomotions', 'gt_bboxes_3d',
                            'gt_labels_3d', 'motion_segmentation',
                            'motion_instance', 'instance_centerness',
                            'instance_offset', 'instance_flow',
                            'has_invalid_frame', 'img_is_valid'
                        ],
                        meta_keys=('filename', 'ori_shape', 'img_shape',
                                   'lidar2img', 'depth2img', 'cam2img',
                                   'pad_shape', 'scale_factor', 'flip',
                                   'pcd_horizontal_flip', 'pcd_vertical_flip',
                                   'box_mode_3d', 'box_type_3d',
                                   'img_norm_cfg', 'pcd_trans', 'sample_idx',
                                   'pcd_scale_factor', 'pcd_rotation',
                                   'pts_filename', 'transformation_3d_flow',
                                   'img_info', 'lidar2ego_rots',
                                   'lidar2ego_trans'))
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False,
            prototype='lift-splat-shoot'),
        test_mode=True,
        box_type_3d='LiDAR',
        receptive_field=2,
        future_frames=3,
        grid_conf=dict(
            xbound=[-51.2, 51.2, 0.8],
            ybound=[-51.2, 51.2, 0.8],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[1.0, 60.0, 1.0]),
        map_grid_conf=dict(
            xbound=[-30.0, 30.0, 0.15],
            ybound=[-15.0, 15.0, 0.15],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[1.0, 60.0, 1.0])),
    test=dict(
        type='RobodriveDataset',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes_infos/robodrive_infos_test.pkl',
        pseudo_bda=True,
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles_MTL',
                using_ego=True,
                data_aug_conf=dict(
                    resize_lim=(0.82, 0.99),
                    final_dim=(512, 1408),
                    rot_lim=(-5.4, 5.4),
                    H=900,
                    W=1600,
                    rand_flip=True,
                    bot_pct_lim=(0.0, 0.22),
                    crop_h=(0.0, 0.0),
                    cams=[
                        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6)),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=[
                            'img_inputs', 'future_egomotions',
                            'has_invalid_frame', 'img_is_valid'
                        ],
                        meta_keys=('filename', 'sample_idx', 'ori_shape',
                                   'img_shape', 'lidar2img', 'depth2img',
                                   'cam2img', 'pad_shape', 'scale_factor',
                                   'flip', 'pcd_horizontal_flip',
                                   'pcd_vertical_flip', 'box_mode_3d',
                                   'box_type_3d', 'img_norm_cfg', 'pcd_trans',
                                   'sample_idx', 'pcd_scale_factor',
                                   'pcd_rotation', 'pts_filename',
                                   'transformation_3d_flow', 'img_info',
                                   'lidar2ego_rots', 'lidar2ego_trans'))
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False,
            prototype='lift-splat-shoot'),
        test_mode=True,
        box_type_3d='LiDAR',
        corruption_root='./data/robodrive-phase2',
        receptive_field=2,
        future_frames=3,
        grid_conf=dict(
            xbound=[-51.2, 51.2, 0.8],
            ybound=[-51.2, 51.2, 0.8],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[1.0, 60.0, 1.0]),
        map_grid_conf=dict(
            xbound=[-30.0, 30.0, 0.15],
            ybound=[-15.0, 15.0, 0.15],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[1.0, 60.0, 1.0])))
evaluation = dict(
    interval=999,
    pipeline=[
        dict(
            type='LoadMultiViewImageFromFiles_MTL',
            using_ego=True,
            data_aug_conf=dict(
                resize_lim=(0.82, 0.99),
                final_dim=(512, 1408),
                rot_lim=(-5.4, 5.4),
                H=900,
                W=1600,
                rand_flip=True,
                bot_pct_lim=(0.0, 0.22),
                crop_h=(0.0, 0.0),
                cams=[
                    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                ],
                Ncams=6)),
        dict(
            type='MultiScaleFlipAug3D',
            img_scale=(1333, 800),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ],
                    with_label=False),
                dict(
                    type='Collect3D',
                    keys=[
                        'img_inputs', 'future_egomotions', 'has_invalid_frame',
                        'img_is_valid'
                    ],
                    meta_keys=('filename', 'sample_idx', 'ori_shape',
                               'img_shape', 'lidar2img', 'depth2img',
                               'cam2img', 'pad_shape', 'scale_factor', 'flip',
                               'pcd_horizontal_flip', 'pcd_vertical_flip',
                               'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                               'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                               'pcd_rotation', 'pts_filename',
                               'transformation_3d_flow', 'img_info',
                               'lidar2ego_rots', 'lidar2ego_trans'))
            ])
    ])
optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 0.0001),
    cyclic_times=1,
    step_ratio_up=0.4)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)
runner = dict(type='EpochBasedRunner', max_epochs=20)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/beverse_fb'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = False
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
sync_bn = True
data_aug_conf = dict(
    resize_lim=(0.82, 0.99),
    final_dim=(512, 1408),
    rot_lim=(-5.4, 5.4),
    H=900,
    W=1600,
    rand_flip=True,
    bot_pct_lim=(0.0, 0.22),
    crop_h=(0.0, 0.0),
    cams=[
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    Ncams=6)
bev_aug_params = dict(
    rot_range=[-0.3925, 0.3925],
    scale_range=[0.95, 1.05],
    trans_std=[0, 0, 0],
    hflip=0.5,
    vflip=0.5)
det_grid_conf = dict(
    xbound=[-51.2, 51.2, 0.8],
    ybound=[-51.2, 51.2, 0.8],
    zbound=[-10.0, 10.0, 20.0],
    dbound=[1.0, 60.0, 1.0])
motion_grid_conf = dict(
    xbound=[-50.0, 50.0, 0.5],
    ybound=[-50.0, 50.0, 0.5],
    zbound=[-10.0, 10.0, 20.0],
    dbound=[1.0, 60.0, 1.0])
map_grid_conf = dict(
    xbound=[-30.0, 30.0, 0.15],
    ybound=[-15.0, 15.0, 0.15],
    zbound=[-10.0, 10.0, 20.0],
    dbound=[1.0, 60.0, 1.0])
grid_conf = dict(
    xbound=[-51.2, 51.2, 0.8],
    ybound=[-51.2, 51.2, 0.8],
    zbound=[-10.0, 10.0, 20.0],
    dbound=[1.0, 60.0, 1.0])
receptive_field = 3
future_frames = 3
future_discount = 0.95
voxel_size = [0.1, 0.1, 0.2]
model = dict(
    type='FBverse',
    img_backbone=dict(
        type='ConvNeXt',
        arch='pico',
        drop_path_rate=0.1,
        layer_scale_init_value=0.0,
        use_grn=True,
        out_indices=(2, 3),
        gap_before_final_norm=False),
    img_neck=dict(type='FPN_LSS', in_channels=768, inverse=True),
    transformer=dict(
        type='TransformerLSS',
        grid_conf=dict(
            xbound=[-51.2, 51.2, 0.8],
            ybound=[-51.2, 51.2, 0.8],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[1.0, 60.0, 1.0]),
        input_dim=(512, 1408),
        numC_input=80,
        numC_Trans=80),
    temporal_model=dict(
        type='Temporal3DConvModel',
        grid_conf=dict(
            xbound=[-51.2, 51.2, 0.8],
            ybound=[-51.2, 51.2, 0.8],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[1.0, 60.0, 1.0]),
        receptive_field=2,
        input_egopose=True,
        in_channels=80,
        input_shape=(128, 128),
        with_skip_connect=True,
        start_out_channels=80),
    pts_bbox_head=dict(
        type='MultiTaskHead',
        in_channels=80,
        out_channels=256,
        grid_conf=dict(
            xbound=[-51.2, 51.2, 0.8],
            ybound=[-51.2, 51.2, 0.8],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[1.0, 60.0, 1.0]),
        det_grid_conf=dict(
            xbound=[-51.2, 51.2, 0.8],
            ybound=[-51.2, 51.2, 0.8],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[1.0, 60.0, 1.0]),
        map_grid_conf=dict(
            xbound=[-30.0, 30.0, 0.15],
            ybound=[-15.0, 15.0, 0.15],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[1.0, 60.0, 1.0]),
        motion_grid_conf=dict(
            xbound=[-50.0, 50.0, 0.5],
            ybound=[-50.0, 50.0, 0.5],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[1.0, 60.0, 1.0]),
        using_ego=True,
        task_enbale=dict({
            '3dod': False,
            'map': True,
            'motion': False
        }),
        task_weights=dict({
            '3dod': 1.0,
            'map': 10.0,
            'motion': 1.0
        }),
        bev_encode_block='Basic',
        cfg_3dod=dict(
            type='CenterHeadv1',
            in_channels=256,
            tasks=[
                dict(num_class=1, class_names=['car']),
                dict(
                    num_class=2, class_names=['truck',
                                              'construction_vehicle']),
                dict(num_class=2, class_names=['bus', 'trailer']),
                dict(num_class=1, class_names=['barrier']),
                dict(num_class=2, class_names=['motorcycle', 'bicycle']),
                dict(num_class=2, class_names=['pedestrian', 'traffic_cone'])
            ],
            common_heads=dict(
                reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
            share_conv_channel=64,
            bbox_coder=dict(
                type='CenterPointBBoxCoder',
                pc_range=[-51.2, -51.2],
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                max_num=500,
                score_threshold=0.1,
                out_size_factor=8,
                voxel_size=[0.1, 0.1],
                code_size=9),
            separate_head=dict(
                type='SeparateHead', init_bias=-2.19, final_kernel=3),
            loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
            loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
            norm_bbox=True),
        cfg_map=dict(
            type='MapHead',
            task_dict=dict(semantic_seg=4),
            in_channels=256,
            class_weights=[1.0, 2.0, 2.0, 2.0],
            semantic_thresh=0.25),
        cfg_motion=dict(
            type='IterativeFlow',
            task_dict=dict(
                segmentation=2,
                instance_center=1,
                instance_offset=2,
                instance_flow=2),
            in_channels=256,
            grid_conf=dict(
                xbound=[-50.0, 50.0, 0.5],
                ybound=[-50.0, 50.0, 0.5],
                zbound=[-10.0, 10.0, 20.0],
                dbound=[1.0, 60.0, 1.0]),
            class_weights=[1.0, 2.0],
            receptive_field=2,
            n_future=3,
            future_discount=0.95,
            using_focal_loss=True,
            prob_latent_dim=32,
            future_dim=6,
            distribution_log_sigmas=[-5.0, 5.0],
            n_gru_blocks=1,
            n_res_layers=3,
            loss_weights=dict(
                loss_motion_seg=5.0,
                loss_motion_centerness=1.0,
                loss_motion_offset=1.0,
                loss_motion_flow=1.0,
                loss_motion_prob=10.0),
            using_spatial_prob=True,
            sample_ignore_mode='past_valid',
            posterior_with_label=False)),
    train_cfg=dict(
        pts=dict(
            point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            grid_size=[1024, 1024, 40],
            voxel_size=[0.1, 0.1, 0.2],
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    test_cfg=dict(
        pts=dict(
            pc_range=[-51.2, -51.2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            pre_max_size=1000,
            post_max_size=83,
            nms_type=[
                'rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'
            ],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[
                1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]
            ])),
    depth_net=dict(
        type='CM_DepthNet',
        in_channels=512,
        context_channels=80,
        downsample=16,
        grid_config=dict(
            x=[-51.2, 51.2, 0.8],
            y=[-51.2, 51.2, 0.8],
            z=[-10.0, 10.0, 20.0],
            depth=[2.0, 42.0, 0.5]),
        depth_channels=80,
        with_cp=True,
        loss_depth_weight=1.0,
        use_dcn=False),
    backward_projection=dict(
        type='BackwardProjection',
        bev_h=128,
        bev_w=128,
        in_channels=80,
        out_channels=80,
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        transformer=dict(
            type='BEVFormer',
            use_cams_embeds=False,
            embed_dims=80,
            encoder=dict(
                type='BevformerEncoder',
                num_layers=1,
                pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                grid_config=dict(
                    x=[-51.2, 51.2, 0.8],
                    y=[-51.2, 51.2, 0.8],
                    z=[-10.0, 10.0, 20.0]),
                data_config=dict(input_size=(512, 1408)),
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerEncoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=80,
                            dropout=0.0,
                            num_levels=1),
                        dict(
                            type='DA_SpatialCrossAttention',
                            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                            dbound=[2.0, 42.0, 0.5],
                            dropout=0.0,
                            deformable_attention=dict(
                                type='DA_MSDeformableAttention',
                                embed_dims=80,
                                num_points=8,
                                num_levels=1),
                            embed_dims=80)
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=80,
                        feedforward_channels=320,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    feedforward_channels=320,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='CustormLearnedPositionalEncoding',
            num_feats=40,
            row_num_embed=128,
            col_num_embed=128)))
data_info_path = 'data/nuscenes_infos/'
val_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_MTL',
        using_ego=True,
        data_aug_conf=dict(
            resize_lim=(0.82, 0.99),
            final_dim=(512, 1408),
            rot_lim=(-5.4, 5.4),
            H=900,
            W=1600,
            rand_flip=True,
            bot_pct_lim=(0.0, 0.22),
            crop_h=(0.0, 0.0),
            cams=[
                'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
            ],
            Ncams=6)),
    dict(
        type='LoadAnnotations3D_MTL',
        with_bbox_3d=True,
        with_label_3d=True,
        with_instance_tokens=True),
    dict(
        type='RasterizeMapVectors',
        map_grid_conf=dict(
            xbound=[-30.0, 30.0, 0.15],
            ybound=[-15.0, 15.0, 0.15],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[1.0, 60.0, 1.0])),
    dict(
        type='ConvertMotionLabels',
        grid_conf=dict(
            xbound=[-50.0, 50.0, 0.5],
            ybound=[-50.0, 50.0, 0.5],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[1.0, 60.0, 1.0]),
        only_vehicle=True),
    dict(type='ObjectValidFilter'),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(
                type='Collect3D',
                keys=[
                    'img_inputs', 'semantic_indices', 'semantic_map',
                    'future_egomotions', 'gt_bboxes_3d', 'gt_labels_3d',
                    'motion_segmentation', 'motion_instance',
                    'instance_centerness', 'instance_offset', 'instance_flow',
                    'has_invalid_frame', 'img_is_valid'
                ],
                meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                           'depth2img', 'cam2img', 'pad_shape', 'scale_factor',
                           'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip',
                           'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                           'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                           'pcd_rotation', 'pts_filename',
                           'transformation_3d_flow', 'img_info',
                           'lidar2ego_rots', 'lidar2ego_trans'))
        ])
]
corruption_root = './data/robodrive-phase2'
corruptions = [
    'brightness', 'dark', 'fog', 'frost', 'snow', 'contrast', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'elastic_transform',
    'color_quant', 'gaussian_noise', 'impulse_noise', 'shot_noise',
    'iso_noise', 'pixelate', 'jpeg_compression'
]
use_checkpoint = True
grid_config = dict(
    x=[-51.2, 51.2, 0.8],
    y=[-51.2, 51.2, 0.8],
    z=[-10.0, 10.0, 20.0],
    depth=[2.0, 42.0, 0.5])
depth_categories = 80
grid_config_bevformer = dict(
    x=[-51.2, 51.2, 0.8], y=[-51.2, 51.2, 0.8], z=[-10.0, 10.0, 20.0])
_dim_ = 512
_pos_dim_ = 40
bev_h_ = 128
bev_w_ = 128
numC_Trans = 80
_num_levels_ = 1
_ffn_dim_ = 320
gpu_ids = range(0, 8)
