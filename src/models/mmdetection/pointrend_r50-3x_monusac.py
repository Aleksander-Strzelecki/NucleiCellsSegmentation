# The new config inherits a base config to highlight the necessary modification
_base_ = '../../../mmdetection/configs/point_rend/point-rend_r50-caffe_fpn_ms-3x_coco.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=4), mask_head=dict(num_classes=4), point_head=dict(num_classes=4)))

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
    optimizer=dict(_delete_=True, type='Adam', lr=0.0003, weight_decay=0.0001))


# optimizer_config=dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + '/val.json')
test_evaluator = val_evaluator

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/point_rend/point_rend_r50_caffe_fpn_mstrain_3x_coco/point_rend_r50_caffe_fpn_mstrain_3x_coco-e0ebb6b7.pth'

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
            'project': 'magisterka',
            'group': 'point_rend_adam'
         },
        )
]


visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='auto',
        interval=1,
        max_keep_ckpts=2))