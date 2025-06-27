import argparse
import os
from PIL import Image
import json
from tqdm import tqdm
import torch
import clip


def main(args):
    if args.dataset == 'coco':
        args.split = 'train'
        data_path = f'data/COCO/{args.split}{args.year}'
        images = os.listdir(data_path)
        image_ids = [int(image.split('_')[-1].split('.')[0]) for image in images]
    elif args.dataset == 'okvqa':
        data_path = f'data/COCO/{args.split}2014'
        target_path = f'data/OK-VQA/OpenEnded_mscoco_{args.split}2014_questions.json'
        images = json.load(open(target_path))['questions']
        image_ids = [int(image['image_id']) for image in images]
    else:
        data_path = f'data/COCO/{args.split}2017'
        target_path = f'data/A-OKVQA/aokvqa_v1p0_{args.split}.json'
        images = json.load(open(target_path))
        image_ids = [int(image['image_id']) for image in images]

    clip_model, feature_extractor = clip.load("ViT-L/14", device='cuda:0', download_root='.cache')
    clip_model.eval()

    bs = args.batch_size
    id_list, feat_list = [], []
    for ids in tqdm(range(0, len(image_ids), bs)):
        image_id = image_ids[ids:ids+bs]
        image_path = [os.path.join(data_path, f'COCO_{args.split}2014_{idx:012d}.jpg') for idx in image_id] if '2014' in data_path \
            else [os.path.join(data_path, f'{idx:012d}.jpg') for idx in image_id]
        image_features = [Image.open(image).convert("RGB") for image in image_path]
        image_features = [feature_extractor(feat) for feat in image_features]
        with torch.no_grad():
            image_features = clip_model.encode_image(torch.stack(image_features).to('cuda:0'))
        id_list.extend(image_id)
        feat_list.append(image_features)

    datastore = {'image_id': id_list, 'image_feature': torch.cat(feat_list)}
    save_path = os.path.join('features', f'coco{args.year[2:]}_train_vitl14_img_features.pt') if args.dataset == 'coco' \
        else os.path.join('features', f'{args.dataset}_{args.split}_vitl14_img_features.pt')
    torch.save(datastore, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument("--dataset", type=str, default="okvqa", choices=['coco', 'okvqa', 'aokvqa'])
    parser.add_argument("--year", type=str, default='2014', choices=['2014', '2017'])
    parser.add_argument("--split", type=str, default="train", choices=['train', 'val'])
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    main(args)
