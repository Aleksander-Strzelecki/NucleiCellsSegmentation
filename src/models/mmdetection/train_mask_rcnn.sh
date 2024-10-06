export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128'
python ../../../mmdetection/tools/train.py ./mask-rcnn_r50-caffe_fpn_ms-poly-3x_monusac.py