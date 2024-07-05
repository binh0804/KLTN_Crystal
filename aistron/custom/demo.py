import json
with open('/media/binh/D/ComputerVision/data/datasets/COCOA/annotations/COCO_amodal_train2014_detectron_no_stuff_aistron.json') as file:
    data = json.load(file)
    # Convert JSON to Python object
    for key in data["annotations"].keys():
        print(key, ":")
        print("\n")