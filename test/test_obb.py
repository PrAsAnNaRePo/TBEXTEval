import os
from ultralytics import YOLO
from PIL import Image, ImageDraw

class OBBModule():
    def __init__(self) -> None:
        self.model = YOLO('18102024.pt')
        # set model parameters
        self.model.overrides['conf'] = 0.35  # NMS confidence threshold
        self.model.overrides['iou'] = 0.45  # NMS IoU threshold
        self.model.overrides['agnostic_nms'] = False  # NMS class-agnostic
        self.model.overrides['max_det'] = 1000  # maximum number of detections per image

    def detect_bbox(self, img, file_name):
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        results = self.model.predict(img)
        print("Number of tables found: ", len(results[0].boxes.xyxy))
        for conf, table_bbox in zip(results[0].boxes.conf.tolist(), results[0].boxes.xyxy):
            print("Confidence: ", conf)
            if conf > 0.26:
                bbox = table_bbox.tolist()
                x1, y1, x2, y2 = bbox[:4]
                draw = ImageDraw.Draw(img)
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            else:
                print("didn't consider the table that has Confidence: ", conf)
        img.save("obb_results-new/" + file_name)

def convert_image_dpi(image_path, target_dpi):
    """Resample image to the target DPI."""
    with Image.open(image_path) as img:
        # Calculate new dimensions based on the target DPI
        width, height = img.size
        new_width = int(width * target_dpi / 275)
        new_height = int(height * target_dpi / 275)
        
        # Resample image
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img.info['dpi'] = (target_dpi, target_dpi)
        return img
    

data_path = 'tbexteval/data/cropp_tool/images/'

obb_module = OBBModule()

for file_name in os.listdir(data_path):
    file_path = data_path + file_name
    img = convert_image_dpi(file_path, 75)
    obb_module.detect_bbox(img, file_name)

