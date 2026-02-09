import numpy as np

import os
import pickle
import click

# python eye_landmark.py --label-path ./../dependencies/datasets/face_landmark/test.txt --pred-path ./../dependencies/datasets/face_landmark/newest/model12_edited_mul-onnx-inference.pkl


@click.command()
@click.option('--label-path', type=str, help='text file')
@click.option('--pred-path', type=str)

def main(
    label_path: str, 
    pred_path: str
) -> None:
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    label = dict()
    for l in lines:
        l = l.strip().split(' ')
        image_name = os.path.basename(l[0])
        target = np.array(l[1:]).astype(np.float32)[36*2:48*2]
        label[image_name] = target

    with open(pred_path, 'rb') as f:
        preds = pickle.load(f)

    index=[0, 9]
    total = 0.
    for k in preds.keys():
        pred = np.array(preds[k][0])
        pred = pred.reshape(12, 2)
        
        target = label[k]
        target = target.reshape(-1, 2)
        norm = np.linalg.norm(target[index[0]] - target[index[1]])      
        nme = np.mean(np.linalg.norm(pred - target, axis=1)) / (norm + 1e-5)
        total += nmeA
        
    print(f"NME: {total / len(preds)}")
    
if __name__=='__main__':
    main()