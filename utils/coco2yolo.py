from ultralytics.data.converter import convert_coco
import os


COCO_SPLIT_PATH = "../data/processed/MoNuSAC_coco_sahi_split"
SAVE_DIR = "../data/processed/MoNuSAC_yolo_sahi_split"
IMAGES_PATH = "../data/processed/MoNuSAC_coco_sahi_split/train_images_224_02"

convert_coco(labels_dir=COCO_SPLIT_PATH, save_dir=SAVE_DIR, use_segments=True)


def create_symbolic_links(src_directory, dest_directory):

    files = os.listdir(src_directory)

    train_files = os.listdir(os.path.join(SAVE_DIR, "labels", "train"))
    for file in train_files:
        dst_train_dir = os.path.join(dest_directory, "train")
        image_name = file[:-4] + ".jpg"
        src_path = os.path.join(src_directory, image_name)
        dest_path = os.path.join(dest_directory, "train", image_name)
        if not os.path.exists(dst_train_dir):
            os.makedirs(dst_train_dir)
        if os.path.isfile(src_path):
            os.symlink(src_path, dest_path)

    val_files = os.listdir(os.path.join(SAVE_DIR, "labels", "val"))
    for file in val_files:
        dst_val_dir = os.path.join(dest_directory, "val")
        image_name = file[:-4] + ".jpg"
        src_path = os.path.join(src_directory, image_name)
        dest_path = os.path.join(dest_directory, "val", image_name)
        if not os.path.exists(dst_val_dir):
            os.makedirs(dst_val_dir)
        if os.path.isfile(src_path):
            os.symlink(src_path, dest_path)

source_directory = IMAGES_PATH
destination_directory = "../data/processed/MoNuSAC_yolo_sahi_split/images"

create_symbolic_links(source_directory, destination_directory)