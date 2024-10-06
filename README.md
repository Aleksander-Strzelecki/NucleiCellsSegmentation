# Segmentation and classification of cell nuclei based on microscopic images using neural networks

This project aims to test various neural network models on the MoNuSAC 2020 challenge dataset.

# Installation

Installation should be performed in a [**Python>=3.8**](https://www.python.org/) environment with [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).

The mmcv submodule has been set up for the potential addition of the cp-cluster algorithm. However, this implementation could not be included due to NVIDIA's reserved rights for the cp-cluster algorithm. If you wish to add this functionality yourself, create the following files in the mmcv submodule:

```
mmcv/ops/csrc/pytorch/cp_cluster.cpp
mmcv/ops/csrc/pytorch/cpu/cp_cluster.cpp
```
For detailed instruction refer to the [mmcv instruction for adding new ops](https://github.com/open-mmlab/mmcv/blob/main/mmcv/ops/csrc/README.md). 

**Note**. If you do not need the cp-cluster implementation, you can simply install the standard mmcv version **2.1.0** using [mmcv install instruction](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)

Use following command in each submodule directory (hover_net, mmcv, mmdetection, sahi, ultralytics, pycococreator):

```
pip install - e .
```

Next, install the project dependencies by running the following command:

```pip install -r requirements.txt```

# Project structure
```
/histopathology_app - Web application for presenting neural network   predictions on histopathology images in *.tif format

/notebooks - Helper notebooks for working with the MoNuSAC dataset

/src - Scripts for training and testing all neural network models used in the project

/utils - Helper scripts for working with the MoNuSAC dataset

```

# Web application

To run the web application, download the pre-trained model weights from this [link](https://drive.google.com/drive/folders/1rNd_juBMiowPzy-L7uwLsfT7Fq0GCdpX?usp=sharing) and place them in the ```histopathology_app/models/weights``` folder. Then, start the application by running the following command inside the ```histopathology_app``` directory:

```
python ./app.py`
```

# Acknowledgments

- https://dagshub.com/chandlerbing65nm/Pan-Cancer-Nuclei-Instance-Segmentation

- R. Verma et al., "MoNuSAC2020: A Multi-Organ Nuclei Segmentation and Classification Challenge," in IEEE Transactions on Medical Imaging, vol. 40, no. 12, pp. 3413-3423, Dec. 2021, doi: 10.1109/TMI.2021.3085712.
keywords: {Annotations;Image segmentation;Tumors;Computer architecture;Training;Task analysis;Semantics;Multi-organ dataset;nucleus classification;computational pathology;instance segmentation;panoptic quality},

- https://monusac-2020.grand-challenge.org/
