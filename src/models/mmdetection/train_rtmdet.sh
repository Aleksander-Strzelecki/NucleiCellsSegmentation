export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128'
python ../../../mmdetection/tools/train.py ./rtmdet-ins_l_8xb16-300e_monusac.py