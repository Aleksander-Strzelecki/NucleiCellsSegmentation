import fiftyone as fo

labels_path = "../data/processed/MoNuSAC_coco_sahi_split/train_224_02.json"
dataset = fo.Dataset.from_dir(data_path="../data/processed/MoNuSAC_coco_sahi_split/train_images_224_02", labels_path=labels_path, dataset_type=fo.types.COCODetectionDataset)

# labels_path = "./dataset/train.json"
# dataset = fo.Dataset.from_dir(data_path="./dataset/train", labels_path=labels_path, dataset_type=fo.types.COCODetectionDataset, max_samples=10)
session = fo.launch_app(dataset)
session.wait()
