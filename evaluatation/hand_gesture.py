from sklearn.metrics import accuracy_score, f1_score

import pickle 
import click
import json

# python hand_gesture.py --json-path ./../dependencies/datasets/pha_handgesture/new_data/labels.json --pred-path ./../dependencies/datasets/pha_handgesture/new_data/pha_hg_0005.pkl

@click.command()
@click.option('--json-path', type=str)
@click.option('--pred-path', type=str)

def main(
    json_path:str,
    pred_path: str
) -> None:    
    
    with open(pred_path, 'rb') as f:
        preds = pickle.load(f)

    res_pred = list()
    for k in preds.keys():
        res_pred.append(preds[k]['label'][0])

    res_gt = list()
    with open(json_path, 'r') as f:
        gt = json.load(f)
    for k in preds.keys():
        res_gt.append(gt[k])
    
    acc = accuracy_score(res_gt, res_pred)
    f1_macro = f1_score(res_gt, res_pred, average='macro')
    f1_micro = f1_score(res_gt, res_pred, average='micro')
    
    print(f'Accuracy: {round(acc, 4)}')
    print(f'F1 macro: {round(f1_macro, 4)}')
    print(f'F1 micro: {round(f1_micro, 4)}')
    
if __name__=='__main__':
    main()