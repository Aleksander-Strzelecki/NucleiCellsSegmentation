import os, subprocess
from PIL import Image
import shutil

MODEL_PATH = "./models/weights/solov2.pth"
CONFIG_PATH = "./models/configs/solov2_r50-1x_monusac.py"
HOVER_NET_WORKDIR = "./models/tmp/hover_net_workdir"
HOVER_NET_OUTPUT = "./models/tmp/hover_net_output"
VISUAL_OUTPUT_PATH = "./models/tmp"



def predict(file_path):
    try:
        image = Image.open(file_path)
        converted_filepath = os.path.join(HOVER_NET_WORKDIR, 'original_image.png')
        os.makedirs(HOVER_NET_WORKDIR, exist_ok=True)
        image.save(converted_filepath)
        print(os.getcwd())
        process = subprocess.Popen(f"python ../hover_net/run_infer.py --batch_size=1 --nr_inference_workers=1 --nr_types=5 --type_info_path=./models/configs/monusac_type.json --model_path=./models/weights/hovernet_fast_monusac_type_tf2pytorch.tar tile --input_dir={HOVER_NET_WORKDIR} --output_dir={HOVER_NET_OUTPUT}", shell=True)
        process.wait()

        shutil.move(os.path.join(HOVER_NET_OUTPUT, 'overlay', 'original_image.png'), os.path.join(VISUAL_OUTPUT_PATH, 'output_image.png'))


    except Exception as e:
        print(f"An error occurred: {e}")

    filepath = os.path.join(VISUAL_OUTPUT_PATH, "output_image.png")
    
    return filepath