from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import os
from PIL import Image

MODEL_PATH = "./models/weights/solov2.pth"
CONFIG_PATH = "./models/configs/solov2_r50-1x_monusac.py"
VISUAL_OUTPUT_PATH = "./models/tmp"

def predict(file_path):
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='mmdet',
        model_path=MODEL_PATH,
        config_path=CONFIG_PATH,
        confidence_threshold=0.4,
        image_size=640,
        device="cpu",
    )

    image = Image.open(file_path)
    converted_filepath = os.path.join(VISUAL_OUTPUT_PATH, 'original_image.png')
    image.save(converted_filepath)
    
    pred = get_sliced_prediction(file_path, detection_model,
    slice_height=224, slice_width=224,
    overlap_height_ratio=0.4, overlap_width_ratio=0.4)


    pred.export_visuals(VISUAL_OUTPUT_PATH, file_name="output_image", hide_labels=True, hide_conf=True)
    filepath = os.path.join(VISUAL_OUTPUT_PATH, "output_image.png")
    
    return filepath
