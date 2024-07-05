export AISTRON_DATASETS=/media/binh/D/ComputerVision/data/datasets

python datasets/prepare_coco.py \
        $AISTRON_DATASETS/COCOA/roboflow/train/_annotations.coco.json
python datasets/prepare_coco.py \
        $AISTRON_DATASETS/COCOA/roboflow/valid/_annotations.coco.json
python datasets/prepare_coco.py \
        $AISTRON_DATASETS/COCOA/roboflow/test/_annotations.coco.json