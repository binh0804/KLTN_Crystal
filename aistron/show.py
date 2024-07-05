import random
import matplotlib.pyplot as plt
import os
#Thiết lập một số hàm cần dùng từ detectron2
import cv2
# Thiết lập Detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# Import một số chức năng cần dùng của Detectron2 như:
# Model Zoo : dùng để tải các tài nguyên hoặc
# tệp huấn luyện sẵn của các mô hình Detectron2
from detectron2 import model_zoo

# DefaultPredictor: Dùng để thiết lập công cụ dự đoán (suy luận)
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from draw_data_myself import Visualizer, ColorMode, _create_text_labels, GenericMask
from aistron.utils.visualizer import AmodalVisualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from aistron.data.datasets.coco_amodal import register_aistron_cocolike_instances
register_aistron_cocolike_instances(
  "my_dataset_train_2", {},
  json_file = "/media/binh/D/ComputerVision/data/datasets/COCOA/annotations/annotations_aistron.json",
  image_root ="/media/binh/D/ComputerVision/data/datasets/COCOA/annotations/")

def display_preprocessed_images(dataset_name, num_samples=5):
    #dataset_dicts = DatasetCatalog.get(dataset_name)
    #selected_dicts = random.sample(dataset_dicts, num_samples)  # Chọn ngẫu nhiên một số ảnh

    for d in DatasetCatalog.get(dataset_name):
        img = cv2.imread(d["file_name"])

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')

        plt.subplot(1, 2, 2)
        plt.imshow(img, cmap='gray')
        plt.title('Preprocessed Image')

        plt.show()
import os
import time
def show_demo_inference_image_from_dataset(dataset_name, num_samples = 1):
  dataset_dicts = DatasetCatalog.get(dataset_name)
  selected_dicts = random.sample(dataset_dicts, num_samples)  # Chọn ngẫu nhiên một số ảnh
  for d in selected_dicts:
    img_origin = cv2.imread(d["file_name"])
    visualizer = Visualizer(img_origin[:, :, ::-1], metadata=test_metadata, scale=0.8)
    amodalVisualizer = AmodalVisualizer(img_origin[:, :, ::-1], metadata=test_metadata, scale=0.8)
    vis = amodalVisualizer.draw_dataset_dict(d, segm_type= "amodal")
    if vis is not None:
      plt.imshow(vis.get_image()[:, :, ::-1])
      plt.show()
test_dataset = "my_dataset_train_2"
test_metadata = MetadataCatalog.get(test_dataset)
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
show_demo_inference_image_from_dataset("my_dataset_train_2", num_samples=5)