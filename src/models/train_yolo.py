import cv2
from ultralytics import YOLO
# from wandb.integration.ultralytics import add_wandb_callback
import wandb

wandb.init(project="magisterka", job_type="yolov8_without_one_bockbone_cnn")

# model = YOLO('yolov8x-seg.yaml')  # build a new model from YAML
# model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8x-seg.yaml').load('weights/yolov8x-seg.pt')  # build from YAML and transfer weights
# model = YOLO('weights/yolov8x-seg.pt')
# model = YOLO('./runs/segment/train13/weights/last.pt')
# add_wandb_callback(model, enable_model_checkpointing=True)

# Train the model
results = model.train(data='../../data/processed/MoNuSAC_yolo_sahi_split/dataset.yaml', epochs=360, imgsz=224, degrees=45, mosaic=0.5, plots=True, freeze=10)

model.val()

wandb.finish()
# model = YOLO('./runs/segment/train8/weights/best.pt')
# results = model("../../data/processed/MoNuSAC_yolo_sahi_split/images/train/image_1_592_0_816_215.jpg")
# annotated_frame = results[0].plot()
# cv2.imshow("image", annotated_frame)
# cv2.waitKey()