import os
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

#
register_coco_instances(
    name ="annotation_cocoa", 
    metadata ={},
    json_file = "/media/binh/D/ComputerVision/data/datasets/COCOA/annotations/instances_default_aistron.json",
    image_root ="/media/binh/D/ComputerVision/data/datasets/COCOA/annotations")
print("Done registering COCOA dataset")
AISTRON_DATASETS = "/media/binh/D/ComputerVision/data/datasets"
cfg = get_cfg()
config_file = "/media/binh/D/ComputerVision/aistron/configs/COCOA/maskrcnn_R101_FPN_cocoa_8ep_bs2.yaml"
cfg.merge_from_file("/media/binh/D/ComputerVision/aistron/configs/COCOA/maskrcnn_R101_FPN_cocoa_8ep_bs2.yaml")
cfg.DATASETS.TRAIN = ("annotation_cocoa",)
cfg.DATASETS.TEST = ("annotation_cocoa",)
command = f"python /media/binh/D/ComputerVision/aistron/tools/train_net.py --config-file {config_file} --num-gpus 1"
os.system(command)