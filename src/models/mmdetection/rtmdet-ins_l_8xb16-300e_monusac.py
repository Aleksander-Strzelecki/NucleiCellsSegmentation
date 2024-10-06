# The new config inherits a base config to highlight the necessary modification
_base_ = '../../../mmdetection/configs/rtmdet/rtmdet-ins_l_8xb32-300e_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(bbox_head=dict(num_classes=4))

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

# max_epochs = 36
# train_cfg = dict(max_epochs=max_epochs)

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


# optimizer_config=dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + '/val.json')
test_evaluator = val_evaluator

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_l_8xb32-300e_coco/rtmdet-ins_l_8xb32-300e_coco_20221124_103237-78d1d652.pth'

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
            'project': 'magisterka',
            'group': 'rtmdet-ins_l'
         },
        )
]


visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='auto',
        # interval=500,
        max_keep_ckpts=2))