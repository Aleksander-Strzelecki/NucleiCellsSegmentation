export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128'
python ../../../mmdetection/tools/train.py ./solov2_r50-3x_AdamW_monusac.py