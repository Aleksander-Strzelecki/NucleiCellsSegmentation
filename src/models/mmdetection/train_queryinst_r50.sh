export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128'
python ../../../mmdetection/tools/train.py ./queryinst_r50_fpn_300-proposals_crop-ms-480-800-3x_adamw_monusac.py