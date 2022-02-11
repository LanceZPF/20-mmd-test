_base_ = [
    'configs/_base_/datasets/coco_detection.py',
]
# model settings
model = dict(
    type='FCOS',
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    #pretrained='open-mmlab://resnest200',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        dcn=dict(
            type="DyXConv",
            K_convs=4,
            ratios=0.25,
            temperature=34,
            deform_groups=4,
            mode="sigmoid",
            dy_deform_w=True,
            dcn=dict(type='DCNv2')),
        stage_with_dcn=(False, False, False, True),
        style='caffe'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        extra_convs_on_inputs=False,  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead_RCNN',
        num_classes=491,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        regress_ranges=((-1, 32), (-1, 64), (-1, 128), (-1, 256), (-1, 1e8)),
        strides=[8, 16, 32, 64, 128],
        #norm_cfg=None,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        norm_on_bbox=True,
        centerness_on_reg=True,
        #dcn_on_last_conv=True,
        #center_sampling=True,
        #conv_bias=True,
        #loss_bbox=dict(type='GIoULoss', loss_weight=1.0)
    ))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100)
img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717],
    #std=[66.13236, 69.130745, 74.77067],
    std=[1, 1, 1],
    to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
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
        img_scale=(512, 512),
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
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
'''
# optimizer
optimizer = dict(
    lr=1e-3, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

lr_config=dict(
    policy='poly',
    power=0.9,
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.01,
    min_lr=5e-5,
    by_epoch=False)
'''
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
# load_from = "../mmd_sella/logs_nfc_s/latest.pth"
# load_from = "./work_dirs/fcos_normal"
load_from = None
#resume_from = "./work_dirs/fcos_3_cfg/latest.pth"
resume_from=None
workflow = [('train', 1)]

optimizer = dict(
    type='SGD',
    lr=1e-3,
    momentum=0.9,
    weight_decay=1e-4,
    paramwise_cfg=dict(bias_lr_mult=2.0, bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# training mode
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    #step=[16, 22, 64])
    step=[45, 48])
total_epochs = 60

'''
# finetune mode
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    #step=[60])
    step=[8,10])
total_epochs = 12
#total_epochs = 100
'''
checkpoint_config = dict(interval=20)
log_config = dict(interval=60, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
evaluation = dict(interval=1, metric='bbox')