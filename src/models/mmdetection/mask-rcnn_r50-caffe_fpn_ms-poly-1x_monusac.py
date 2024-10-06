# The new config inherits a base config to highlight the necessary modification
_base_ = '../../../mmdetection/configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=4), mask_head=dict(num_classes=4)))

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

max_epochs = 36
train_cfg = dict(max_epochs=max_epochs)

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

param_scheduler = [
    dict(
        type='LinearLR',  # Use linear learning rate warmup
        start_factor=0.001, # Coefficient for learning rate warmup
        by_epoch=False,  # Update the learning rate during warmup at each iteration
        begin=0,  # Starting from the first iteration
        end=3000),  # End at the 500th iteration
    dict(
        type='MultiStepLR',  # Use multi-step learning rate strategy during training
        by_epoch=True,  # Update the learning rate at each epoch
        begin=0,   # Starting from the first epoch
        end=12,  # Ending at the 12th epoch
        milestones=[8, 11],  # Learning rate decay at which epochs
        gamma=0.1)  # Learning rate decay coefficient
]


# optimizer_config=dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + '/val.json')
test_evaluator = val_evaluator

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
            'project': 'magisterka',
            'group': 'maskrcnn-r50-fpn-1x-coco'
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