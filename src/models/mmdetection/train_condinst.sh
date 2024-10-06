export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128'
python ../../../mmdetection/tools/train.py ./condinst_r50_fpn_ms-poly-90k_monusac_instance.py