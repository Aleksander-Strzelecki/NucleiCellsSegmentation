import json

def filter_annotations(coco_json_path, output_json_path, area_threshold=10):
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    new_annotations = []
    for annotation in coco_data['annotations']:
        area = annotation['area']
        if area >= area_threshold:
            new_annotations.append(annotation)
        else:
            print("Area == {}".format(area))

    coco_data['annotations'] = new_annotations

    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f)

coco_json_path = '../data/processed/MoNuSAC_coco_sahi_split/train.json'
output_json_path = '../data/processed/MoNuSAC_coco_sahi_split/tmp.json'
filter_annotations(coco_json_path, output_json_path, area_threshold=10)
