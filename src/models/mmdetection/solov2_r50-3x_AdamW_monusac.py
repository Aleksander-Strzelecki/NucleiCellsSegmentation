# The new config inherits a base config to highlight the necessary modification
_base_ = '../../../mmdetection/configs/solov2/solov2_r50_fpn_ms-3x_coco.py'

work_dir = "../../../mmdetection/work_dir"
# resume=True
# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    mask_head=dict(
        num_classes=4))

# Modify dataset related settings
data_root = '../../../data/processed/MoNuSAC_coco_sahi_split'
metainfo = {
    'classes': ('Epithelial', 'Lymphocyte', 'Macrophage', 'Neutrophil', ),
    'palette': [
        (220, 20, 60),
        (220, 0, 0),
        (0, 220, 0),
        (0, 0, 220),
    ]
}
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train_filtered.json',
        data_prefix=dict(img='train_images_224_02')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val_filtered.json',
        data_prefix=dict(img='train_images_224_02')))
test_dataloader = val_dataloader

optim_wrapper = dict(
    type='OptimWrapper',
    # optimizer
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),

    # Parameter-level learning rate and weight decay settings
    paramwise_cfg=dict(
        _delete_=True,
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
        },
        norm_decay_mult=0.0),

    # gradient clipping
    clip_grad=dict(_delete_=True, max_norm=0.01, norm_type=2))


# optimizer_config=dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + '/val.json')
test_evaluator = val_evaluator

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_r50_fpn_3x_coco/solov2_r50_fpn_3x_coco_20220512_125856-fed092d4.pth'

vis_backends = [
    dict(type='LocalVisBackend'),
    # dict(type='TensorboardVisBackend'),
    # dict(type='WandbVisBackend',
    #      init_kwargs={
    #         'project': 'magisterka',
    #         'group': 'solov2_AdamW'
    #      },
    #     )
]


visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='auto',
        interval=1,
        max_keep_ckpts=2))