export AISTRON_DATASETS=/media/binh/D/ComputerVision/data/datasets

python datasets/prepare_coco.py \
        $AISTRON_DATASETS/COCOA/annotations/instances_default.json \

# python datasets/prepare_coco.py \
#         $AISTRON_DATASETS/coco/annotations/instances_train2017.json \
