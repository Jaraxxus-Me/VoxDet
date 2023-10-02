dataset_type = 'ZidDataset'
data_root = 'data/ZiD/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadP1Info'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='ImagesToTensor', keys=['rgb', 'mask', 'depth']),
    dict(type='Collect', keys=['img', 'rgb', 'mask', 'depth', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadP1Info'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ImagesToTensor', keys=['rgb', 'mask', 'depth']),
            dict(type='Collect', keys=['img', 'rgb', 'mask', 'depth']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        p1_path=data_root + 'P1/',
        ann_file=data_root + 'P2/train_annotations.json',
        img_prefix=data_root + '/P2/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        p1_path=data_root + 'P1/',
        ann_file=data_root + 'P2/val_annotations.json',
        img_prefix=data_root + '/P2/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
