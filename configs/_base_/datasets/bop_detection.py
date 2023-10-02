dataset_type = 'BopDataset'
data_root = 'data/BOP/'
dataset = 'lmo' # 'lmo' 'ycbv'
scene_id = 2 # 'lmo': 2 || 'ycbv': 48 19 50 51 52 53 54 55 56 57 58 59
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
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
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    test=dict(
        type=dataset_type,
        ann_file=data_root + dataset + '/test/{:06}/scene_gt_coco.json'.format(scene_id),
        img_prefix=data_root + dataset + '/test/{:06}'.format(scene_id),
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
