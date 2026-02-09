import os
import shutil
import cv2
import pickle
import json
import random
from colorama import Fore
from .. import utils
from .image_cls import *

class PHACls(ImageClassification):
    def __init__(self, *args, image_path, annotations_file, download=False, num_frames=None, name='phacls', **kwargs):
        # super().__init__()

        if download:
            raise Exception("No download support.")

        assert annotations_file.endswith('json')
        with open(annotations_file, 'r') as f:
            data = json.load(f)
        
        self.imgs = list()
        self.labels = list()

        for k, v in data.items():
            this_image_path = os.path.join(image_path, k)
            assert os.path.exists(this_image_path)
            self.imgs.append(this_image_path)
            self.labels.append(v)
        
        shuffle = kwargs.get('shuffle', False)
        if shuffle:
            seed = 314 # to reproduce calibration process
            random.seed(seed)
            random.shuffle(self.imgs)
            random.seed(seed)
            random.shuffle(self.labels)

        self.num_frames = len(self.imgs) if num_frames is None else num_frames
        self.kwargs = kwargs

        info = dict(description='Age PHA', url='', version='1.0',
                        year='2024',
                        contributor='',
                        date_created='')
        # categories = [
        #     {'id': 0, 'name': '0-2', 'wnid': '0'},
        #     {'id': 1, 'name': '3-9', 'wnid': '1'},
        #     {'id': 2, 'name': '10-19', 'wnid': '2'},
        #     {'id': 3, 'name': '20-29', 'wnid': '3'},
        #     {'id': 4, 'name': '30-39', 'wnid': '4'},
        #     {'id': 5, 'name': '40-49', 'wnid': '5'},
        #     {'id': 6, 'name': '50-59', 'wnid': '6'},
        #     {'id': 7, 'name': '60-69', 'wnid': '7'},
        #     {'id': 8, 'name': '70++', 'wnid': '8'},
        #     ]
        # categories = list()
        # for i in range(1000):
        #     this_content = {'id': i, 'name': f'{i}', 'wnid': str(i)}
        #     categories.append(this_content)
        category_path = os.path.join(os.path.dirname(annotations_file), 'label_map.json')
        with open(category_path, 'r') as f:
            json_category = json.load(f)
        
        categories = list()
        for k, id in json_category.items():
            this_category = {
                'id': id, 'name': k, 'wnid': id
            }
            categories.append(this_category)
        
        
        self.dataset_store = dict(info=info, categories=categories)
        self.kwargs['dataset_info'] = self.get_dataset_info()

        super().initialize()

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx, **kwargs):
        with_label = kwargs.get('with_label', False)
        if with_label:
            return self.imgs[idx], self.labels[idx]
        else:
            return self.imgs[idx]
    
    def evaluate(self, predictions, **kwargs):
        metric_tracker = utils.AverageMeter(name='accuracy_top1%')
        num_frames = min(self.num_frames, len(predictions))
        for n in range(num_frames):
            gt_label = self.labels[n]
            # print('\n\n\n\n')
            # print(predictions)
            if isinstance(predictions[n], list):
                pred = predictions[n][0]
            else:
                pred = predictions[n]
            accuracy = self.classification_accuracy(pred, gt_label, **kwargs)
            metric_tracker.update(accuracy)
        return {metric_tracker.name:metric_tracker.avg}

    def get_notice(self):
        notice = f'{Fore.YELLOW}' \
                 f'\nPHA Dataset for classification'  \
                 f'{Fore.RESET}\n'
        return notice

    def get_dataset_info(self):
        if self.dataset_store is None:
            return None
        #
        # return only info and categories for now as the whole thing could be quite large.
        dataset_store = dict()
        for key in ('info', 'categories'):
            if key in self.dataset_store.keys():
                dataset_store.update({key: self.dataset_store[key]})
            #
        #
        dataset_store.update(dict(color_map=self.get_color_map(num_classes=len(self.dataset_store['categories']))))        
        return dataset_store