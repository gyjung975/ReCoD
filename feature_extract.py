"""
pre-computing features of datastore
"""
import argparse
import os
from PIL import Image
import json
import torch
import clip


def main(args):
    if args.dataset == 'COCO':  # coco14_train_vitl14_img_features.pt, coco17_train_vitl14_img_features.pt
        args.split = 'train'
        data_path = f'../NAS/0datasets/COCO/{args.split}{args.year}'
        images = os.listdir(data_path)
        image_ids = [int(image.split('_')[-1].split('.')[0]) for image in images]
    elif args.dataset == 'OK-VQA':  # okvqa_train_vitl14_img_features.pt, okvqa_val_vitl14_img_features.pt
        data_path = f'../NAS/0datasets/COCO/{args.split}2014'
        target_path = f'../NAS/0datasets/OK-VQA/OpenEnded_mscoco_{args.split}2014_questions.json'
        images = json.load(open(target_path))['questions']
        image_ids = [int(image['image_id']) for image in images]
    else:       # aokvqa_train_vitl14_img_features.pt, aokvqa_val_vitl14_img_features.pt
        data_path = f'../NAS/0datasets/COCO/{args.split}2017'
        target_path = f'../NAS/0datasets/A-OKVQA/aokvqa_v1p0_{args.split}.json'
        images = json.load(open(target_path))
        image_ids = [int(image['image_id']) for image in images]

    clip_model, feature_extractor = clip.load("ViT-L/14", device='cuda:0', download_root='.cache')
    clip_model.eval()

    bs = 64
    id_list, feat_list = [], []
    for ids in range(0, len(image_ids), bs):
        image_id = image_ids[ids:ids+bs]
        image_path = [os.path.join(data_path, f'COCO_{args.split}2014_{idx:012d}.jpg') for idx in image_id] if '2014' in data_path \
            else [os.path.join(data_path, f'{idx:012d}.jpg') for idx in image_id]
        image_features = [Image.open(image).convert("RGB") for image in image_path]
        image_features = [feature_extractor(feat) for feat in image_features]
        with torch.no_grad():
            image_features = clip_model.encode_image(torch.stack(image_features))
        id_list.extend(image_id)
        feat_list.append(image_features)
    datastore = {'image_id': id_list, 'image_feature': torch.cat(feat_list)}

    if args.dataset == 'COCO': torch.save(datastore, os.path.join('features', f'coco{args.year[2:]}_train_vitl14_img_features.pt'))
    elif args.dataset == 'OK-VQA': torch.save(datastore, os.path.join('features', f'okvqa_{args.split}_vitl14_img_features.pt'))
    else: torch.save(datastore, os.path.join('features', f'aokvqa_{args.split}_vitl14_img_features.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument("--dataset", type=str, default="COCO", choices=['COCO', 'OK-VQA', 'A-OKVQA'])
    parser.add_argument("--year", type=str, default='2014', choices=['2014', '2017'])
    parser.add_argument("--split", type=str, default="train", choices=['train', 'val'])
    args = parser.parse_args()

    main(args)
