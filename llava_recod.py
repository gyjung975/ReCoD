import os
import random
import argparse
import json
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import clip

from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.eval.run_llava import batch_inference
import template


def save_config(args: argparse.Namespace):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.save_path, "args.json")
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile)


def caption_processing(captions):
    captions = [cap.replace('The scene features', '').strip() for cap in captions]
    captions = [cap.replace('The image is', '').strip() for cap in captions]
    captions = [cap.replace('The image displays', '').strip() for cap in captions]
    captions = [cap.replace('The image features', '').strip() for cap in captions]
    captions = [cap.replace('The image captures', '').strip() for cap in captions]
    captions = [cap.replace('The image depicts', '').strip() for cap in captions]
    captions = [cap.replace('The image shows', '').strip() for cap in captions]
    captions = [cap.replace('The image showcases', '').strip() for cap in captions]
    captions = [cap.replace('The image portrays', '').strip() for cap in captions]
    captions = [cap.replace('This scene features', '').strip() for cap in captions]
    captions = [cap.replace('This image is', '').strip() for cap in captions]
    captions = [cap.replace('This image displays', '').strip() for cap in captions]
    captions = [cap.replace('This image features', '').strip() for cap in captions]
    captions = [cap.replace('This image captures', '').strip() for cap in captions]
    captions = [cap.replace('This image depicts', '').strip() for cap in captions]
    captions = [cap.replace('This image shows', '').strip() for cap in captions]
    captions = [cap.replace('This image showcases', '').strip() for cap in captions]
    captions = [cap.replace('This image portrays', '').strip() for cap in captions]
    captions = [cap.replace('In the image,', '').strip() for cap in captions]
    captions = [cap.replace('In the scene,', '').strip() for cap in captions]
    captions = [cap.replace('In this image,', '').strip() for cap in captions]
    captions = [cap.replace('In this scene,', '').strip() for cap in captions]
    return captions


def load_data(args):
    anno_path = os.path.join(args.data_dir, "annotations_trainval2014/captions_train2014.json") if args.dataset == 'okvqa' \
        else os.path.join(args.data_dir, "annotations_trainval2017/captions_train2017.json")

    print('*** preparing datastore')
    annotations = json.load(open(anno_path))['annotations']
    data = {}
    for item in tqdm(annotations):
        if item['image_id'] in data: data[item['image_id']].append(item['caption'])
        else:                        data[item['image_id']] = [item['caption']]
    return data


@torch.no_grad()
def image_retrieval(caption, clip_model, exclude_idx):
    retrieval_score = []
    for cap in caption:
        cap_feature = clip.tokenize(cap, truncate=True).to('cuda:0')
        cap_feature = clip_model.encode_text(cap_feature)
        cap_feature /= cap_feature.norm(dim=-1, keepdim=True)
        similarity = torch.mean(cap_feature @ RETRIEVAL_IMG_FEATURES.t(), dim=0, keepdim=True)
        retrieval_score.append(similarity)

    retrieval_score = torch.cat(retrieval_score, dim=0)

    for i in range(len(caption)):
        retrieval_score[i, exclude_idx[i]] = 0

    retrieved_img_index = np.asarray(torch.argmax(retrieval_score.cpu(), dim=-1))
    retrieved_img_id = can_image_ids[retrieved_img_index]
    return retrieved_img_index, retrieved_img_id


@torch.no_grad()
def main(args):
    clip_model, _ = clip.load("ViT-L/14", device='cuda:0', download_root='.cache')
    clip_model.eval()

    model_name = get_model_name_from_path(args.model_path)
    llava_tokenizer, llava_model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)

    target_datastore_file = '{}_{}_vitl14_img_features.pt'.format(args.dataset, args.split)
    target_datastore = torch.load(os.path.join('features', target_datastore_file))
    image_ids = np.asarray(list(set(target_datastore['image_id'])))

    init_path = f'diff/{args.dataset}/llava_base_{args.split}/llava_base_{args.split}.json'
    init_file = json.load(open(init_path))
    init_captions = {item['image_id']: item['new_captions'][0] for item in init_file}

    retrieval_store = load_data(args)
    dataset = []

    save_config(args)
    for idx in tqdm(range(0, len(image_ids), args.batch_size)):
        target_ids = image_ids[idx:idx+args.batch_size]
        target_path = [args.data_dir + f"{args.split}2014/COCO_{args.split}2014_{target_id:012d}.jpg" for target_id in target_ids] if args.dataset != 'aokvqa' \
            else [args.data_dir + f"{args.split}2017/{target_id:012d}.jpg" for target_id in target_ids]
        target_image = [Image.open(target).convert("RGB") for target in target_path]

        caption = [init_captions[target_idx] for target_idx in target_ids]
        datum = [{'image_id': int(target_ids[d]),
                  'init_caption': caption[d],
                  'retrieved': []}
                 for d in range(len(target_ids))]

        exclude_ids, difference_captions = [[] for _ in target_ids], [[] for _ in target_ids]
        for e in range(len(target_ids)):
            difference_captions[e].append(caption[e])

        caption = [cap.split('. ') for cap in caption]
        for loop in range(args.loop):
            retrieved_img_index, retrieved_image_id = image_retrieval(caption, clip_model, exclude_ids)
            for e in range(len(target_ids)):
                exclude_ids[e].append(retrieved_img_index[e])

            retrieved_image_caption = [retrieval_store[retrieved_id] for retrieved_id in retrieved_image_id]
            refer_caps = [[ric[ri] for ri in random.sample(list(range(5)), k=2)] for ric in retrieved_image_caption]

            differences = batch_inference(
                target_image=target_image, model_name=model_name,
                tokenizer=llava_tokenizer, model=llava_model, image_processor=image_processor,
                text=refer_caps, args=args)
            differences = caption_processing(differences)
            splited = [diff.split('. ') for diff in differences]
            for c in range(len(caption)):
                caption[c] += splited[c]

            for i in range(len(target_path)):
                retri = {'id': int(retrieved_image_id[i]),
                         'captions': retrieved_image_caption[i],
                         'used_captions': refer_caps[i],
                         'differences': differences[i]}
                datum[i]['retrieved'].append(retri)

            for e in range(len(target_path)):
                difference_captions[e].append(differences[e])

        for d in range(len(datum)):
            datum[d]['new_captions'] = difference_captions[d]
        dataset += datum

    with open(os.path.join(args.save_path, args.save_name), 'w') as outfile:
        json.dump(dataset, outfile, indent=4)

    gen_infer_file(os.path.join(args.save_path, args.save_name), args)


def gen_infer_file(raw_file, args):
    if args.dataset == 'okvqa':
        question = json.load(open(f"../NAS/0datasets/OK-VQA/OpenEnded_mscoco_{args.split}2014_questions.json"))['questions']
        answer = json.load(open(f"../NAS/0datasets/OK-VQA/mscoco_{args.split}2014_annotations.json"))['annotations']
        caption = json.load(open(raw_file))

        loop_dict = [{} for _ in range(0, args.loop+1)]
        for item in caption:
            for idx, l in enumerate(range(0, args.loop+1)):
                loop_dict[idx][item['image_id']] = item['new_captions'][:l+1]

        for idx, l in enumerate(range(0, args.loop+1)):
            for q, a in zip(question, answer):
                q['answers'] = a['answers']
                loop_can = loop_dict[idx][q['image_id']]
                q['new_captions'] = loop_can
            json.dump(question, open(f"diff/okvqa/llava_recod_{args.split}/llava_recod_{args.split}_total{l+1}.json", "w"), indent=4)
    else:
        aokvqa = json.load(open(f"../NAS/0datasets/A-OKVQA/aokvqa_v1p0_{args.split}.json"))
        caption = json.load(open(raw_file))

        loop_dict = [{} for _ in range(0, args.loop+1)]
        for item in caption:
            for idx, l in enumerate(range(0, args.loop+1)):
                loop_dict[idx][item['image_id']] = item['new_captions'][:l+1]

        for idx, l in enumerate(range(0, args.loop+1)):
            for d in aokvqa:
                loop_can = loop_dict[idx][d['image_id']]
                d['new_captions'] = loop_can
            json.dump(aokvqa, open(f"diff/aokvqa/llava_recod_{args.split}/llava_recod_{args.split}_total{l+1}.json", "w"), indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--model_path", type=str, default='liuhaotian/llava-v1.5-13b')
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--conv_mode", type=str, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.7)
    parser.add_argument("--query", type=str, default='Describe the image.')
    parser.add_argument("--dataset", type=str, default='okvqa')
    parser.add_argument("--split", type=str, default='val')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--loop", type=int, default=10)
    args = parser.parse_args()

    args.query = template.caption_based
    args.loop = args.loop - 1       # first caption is generated by LLaVA without comparison

    coco = 14 if args.dataset != 'aokvqa' else 17
    datastore_file = 'coco{}_train_vitl14_img_features.pt'.format(coco)
    datastore = torch.load(os.path.join('features', datastore_file))
    can_image_ids = np.asarray(datastore['image_id'])
    can_image_features = datastore['image_feature']

    RETRIEVAL_IMG_FEATURES = can_image_features.to('cuda:0')
    RETRIEVAL_IMG_FEATURES /= RETRIEVAL_IMG_FEATURES.norm(dim=-1, keepdim=True)

    args.save_path = f'diff/{args.dataset}/llava_recod_{args.split}'
    args.save_name = f'llava_recod_{args.split}.json'
    print(os.path.join(args.save_path, args.save_name))

    os.makedirs(args.save_path, exist_ok=True)
    os.system("cp llava_recod.py %s/" % args.save_path)
    main(args)
