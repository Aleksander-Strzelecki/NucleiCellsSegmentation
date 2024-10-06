from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import numpy as np
from scipy.io import savemat
import os
import time

TESTS_IMAGES_PATH = "../../../data/raw/MoNuSAC/MoNuSAC Testing Data and Annotations/"
OUTPUT_PATH = "../../../data/processed/queryinst/trash"
VISUAL_OUTPUT_PATH = "../../../data/processed/queryinst/trash2"
ID_TO_CLASS = {0: "Epithelial", 1: "Lymphocyte", 2: "Macrophage", 3: "Neutrophil"}

model_path = "../../../histopathology_app/models/weights/queryinst.pth"
config_path = "../../../histopathology_app/models/configs/queryinst_r50_fpn_300-proposals_crop-ms-480-800-3x_monusac.py"

detection_model = AutoDetectionModel.from_pretrained(
    model_type='mmdet',
    model_path=model_path,
    config_path=config_path,
    confidence_threshold=0.4,
    image_size=640,
    device="cuda:0", # or 'cuda:0'
)

total_time = 0.0
num_calls = 0

for patient in os.listdir(TESTS_IMAGES_PATH):
    patient_path = os.path.join(TESTS_IMAGES_PATH, patient)
    for subdir, dirs, files in os.walk(patient_path):
        for file in files:
            if file.lower().endswith('.tif'):
                file_path = os.path.join(subdir, file)
                
                start_time = time.time()
                pred = get_sliced_prediction(file_path, detection_model,
                slice_height=224, slice_width=224,
                overlap_height_ratio=0.4, overlap_width_ratio=0.4)
                end_time = time.time()
                elapsed_time = end_time - start_time
                total_time += elapsed_time
                num_calls += 1
                average_time = total_time / num_calls
                print(f"Average time for get_sliced_prediction: {average_time:.4f} seconds")

                np_mask_list = []
                for i in range(4):
                    np_mask_list.append(np.zeros(pred.object_prediction_list[0].mask.bool_mask.shape))

                instance_counter = 1
                for object_prediction in pred.object_prediction_list:
                    np_mask_list[object_prediction.category.id][object_prediction.mask.bool_mask] = instance_counter
                    instance_counter += 1

                for i in range(4):
                    output_mask_dir = os.path.join(OUTPUT_PATH, patient, file[:-4], ID_TO_CLASS[i])
                    output_mask_path = os.path.join(output_mask_dir, "mask.mat")
                    if len(np.unique(np_mask_list[i])) > 1:
                        os.makedirs(output_mask_dir, exist_ok=True)
                        savemat(output_mask_path, {'n_ary_mask': np_mask_list[i]})

                pred.export_visuals(VISUAL_OUTPUT_PATH, file_name=file[:-4], hide_labels=True, hide_conf=True)

if num_calls > 0:
    average_time = total_time / num_calls
    print(f"Average time for get_sliced_prediction: {average_time:.4f} seconds")
else:
    print("No calls to get_sliced_prediction were made.")