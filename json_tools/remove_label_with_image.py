'''
This script is to remove specify classes from coco json annotations

- Arguments:
    + image-path: the image directory.
    + image-save-path: where to save images.
    + json-path: the json path that need to be modified.
    + save-json-path: where to save new json
    + class-idx: class idx to remove


python remove_label_with_image.py \
--image-path /home/edgeai/code/edgeai-benchmark/dependencies/pha_2_datasets/detection/od_10_classes/images \
--json-path /home/edgeai/code/edgeai-benchmark/dependencies/pha_2_datasets/detection/od_10_classes/annotations/final_v1_rough_w_buckle.json \
--image-save-path  /home/edgeai/code/edgeai-benchmark/dependencies/pha_2_datasets/detection/od_7classes/images \
--save-json-path /home/edgeai/code/edgeai-benchmark/dependencies/pha_2_datasets/detection/od_7classes/annotations/test.json \
--class-idx 2 8 9



'''
import os
import json
import argparse

from pprint import pprint
from copy import deepcopy as dp
from shutil import copyfile
from natsort import natsorted
def main(args):
    image_path = args.image_path
    image_save_path = args.image_save_path
    json_path = args.json_path
    save_path = args.save_json_path
    class_idx = args.class_idx
    
    assert os.path.realpath(json_path) != os.path.realpath(save_path), \
        f"Json path and save path are identical, continue will override the original json, if you wish to continue, comment this assertion"

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    pprint(data['categories'])
    print(f'\nRemoving class id: {args.class_idx}')
    print(f"Number of bboxes before process: {len(data['annotations'])}")
    print(f"Number of images before process: {len(data['images'])}")
    
    # process categories
    new_categories = list()
    new_id = 1
    old2new = dict()
    for cate in data['categories']:
        if cate['id'] not in class_idx:
            old2new[cate['id']] = new_id
            cate['id'] = new_id
            new_id += 1
            
            new_categories.append(dp(cate))
    
    # process annotations
    new_annotation = list()
    new_annot_id = 1

    sorted_data_anno = natsorted(data['annotations'], key=lambda x: x["image_id"])

    for annot in sorted_data_anno:
        if annot['category_id'] in class_idx:
            continue
        
        annot['id'] = new_annot_id
        annot['category_id'] = old2new[annot['category_id']]
        new_annot_id += 1
        
        new_annotation.append(dp(annot))
    
    # process images
    used_image = set()
    for annot in new_annotation:
        used_image.add(annot['image_id'])
    
    old2new_image = dict()
    new_image_id = 1
    new_images = list()
    remain_image = list()

    sorted_data_file = natsorted(data['images'], key=lambda x: x["file_name"])

    for img in sorted_data_file:
        if img['id'] in used_image:
            remain_image.append(img['file_name'])
            old2new_image[img['id']] = new_image_id
            img['id'] = new_image_id
            new_image_id += 1
            
            new_images.append(dp(img))

    # fix annotations again to match with new image id
    for annot in new_annotation:
        annot['image_id'] = old2new_image[annot['image_id']]
    
    # copy image to new dir
    if image_save_path is not None:
        for img_name in remain_image:
            copyfile(
                os.path.join(image_path, img_name),
                os.path.join(image_save_path, img_name),
            )
        
    data['annotations'] = new_annotation
    data['categories'] = new_categories
    data['images'] = new_images
    
    print(f"Number of bboxes after process: {len(data['annotations'])}")
    print(f"Number of images after process: {len(data['images'])}")
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)    
    
    

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--image-path",
        type=str,
        help='image path'
    )
    args.add_argument(
        "--image-save-path",
        type=str,
        default=None,
        help='where to save new image path'
    )
    args.add_argument(
        "--save-json-path",
        type=str,
        help='where to save new json'
    )
    args.add_argument(
        "--json-path",
        type=str,
        help='json path'
    )
    args.add_argument(
        "--class-idx",
        nargs='*',
        default=None,
        type=int,
        help="which class should be remove"
    )
    return args.parse_args()

if __name__=='__main__':
    main(get_args())