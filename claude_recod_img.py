import os
import argparse
import json
import numpy as np
import torch
from tqdm import tqdm
import clip
import template
import claude


def save_config(args: argparse.Namespace):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.save_path, "args.json")
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile)


@torch.no_grad()
def image_retrieval(caption, clip_model, exclude_idx):
    cap_feature = clip.tokenize(caption, truncate=True).to('cuda:0')
    cap_feature = clip_model.encode_text(cap_feature)
    cap_feature /= cap_feature.norm(dim=-1, keepdim=True)
    similarity = torch.mean(cap_feature @ RETRIEVAL_IMG_FEATURES.t(), dim=0)

    for eid in exclude_idx:
        similarity[eid] = 0
    retrieved_img_index = torch.argmax(similarity, dim=-1).item()
    retrieved_img_id = can_image_ids[retrieved_img_index].item()
    return retrieved_img_index, retrieved_img_id


@torch.no_grad()
def main(args):
    clip_model, _ = clip.load("ViT-L/14", device='cuda:0', download_root='.cache')
    clip_model.eval()

    target_datastore_file = '{}_{}_vitl14_img_features.pt'.format(args.dataset, args.split)
    target_datastore = torch.load(os.path.join('features', target_datastore_file))
    image_ids = np.asarray(list(set(target_datastore['image_id'])))

    init_path = f'diff/{args.dataset}/claude_base_{args.split}/claude_base_{args.split}.json'
    init_file = json.load(open(init_path))
    init_captions = {item['image_id']: item['new_captions'][0] for item in init_file}

    dataset = []

    save_config(args)
    pbar = tqdm(total=len(image_ids))
    for idx, target_idx in enumerate(image_ids):
        target_path = args.data_dir + f"{args.split}2014/COCO_{args.split}2014_{target_idx:012d}.jpg" if args.dataset != 'aokvqa' \
            else args.data_dir + f"{args.split}2017/{target_idx:012d}.jpg"

        caption = init_captions[target_idx]
        datum = {'image_id': target_idx.item(),
                 'init_caption': caption,
                 'retrieved': []}

        exclude_ids, difference_captions = [], []
        difference_captions.append(caption)

        caption = caption.split('. ')
        for loop in range(args.loop):
            retrieved_img_index, retrieved_image_id = image_retrieval(caption, clip_model, exclude_ids)
            retrieved_path = args.data_dir + f"train2014/COCO_train2014_{retrieved_image_id:012d}.jpg" if args.dataset != 'aokvqa' \
                else args.data_dir + f"train2017/{retrieved_image_id:012d}.jpg"
            exclude_ids.append(retrieved_img_index)

            response = claude.two_images([target_path, retrieved_path], ["image/jpeg"]*2, args.query, args.config)
            differences = response.content[0].text.strip()
            splited = differences.split('. ')

            retri = {'id': int(retrieved_image_id),
                     'used_image_id': int(retrieved_image_id),
                     'differences': differences}
            datum['retrieved'].append(retri)

            difference_captions.append(differences)
            caption += [cap for cap in splited if len(cap.split(' ')) >= 5]

        datum['new_captions'] = difference_captions
        dataset.append(datum)
        pbar.update(1)

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
            json.dump(question, open(f"diff/okvqa/claude_recod_img_{args.split}/claude_recod_img_{args.split}_total{l+1}.json", "w"), indent=4)
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
            json.dump(aokvqa, open(f"diff/aokvqa/claude_recod_img_{args.split}/claude_recod_img_{args.split}_total{l+1}.json", "w"), indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--model", type=str, default="claude-3-haiku-20240307")
    parser.add_argument("--max_new_tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--query", type=str, default='Describe the image.')
    parser.add_argument("--dataset", type=str, default='okvqa')
    parser.add_argument("--split", type=str, default='val')
    parser.add_argument("--loop", type=int, default=10)
    args = parser.parse_args()

    args.config = {
        'model': args.model,
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature,
    }
    args.query = template.image_based
    args.loop = args.loop - 1       # first caption is generated by Claude3 without comparison

    # Load datastore: coco2014 train split for OK-VQA, coco2017 train split for A-OKVQA
    coco = 14 if args.dataset != 'aokvqa' else 17
    datastore_file = 'coco{}_train_vitl14_img_features.pt'.format(coco)
    datastore = torch.load(os.path.join('features', datastore_file))
    can_image_ids = np.asarray(datastore['image_id'])
    can_image_features = datastore['image_feature']

    RETRIEVAL_IMG_FEATURES = can_image_features.to('cuda:0')
    RETRIEVAL_IMG_FEATURES /= RETRIEVAL_IMG_FEATURES.norm(dim=-1, keepdim=True)

    args.save_path = f'diff/{args.dataset}/claude_recod_img_{args.split}'
    args.save_name = f'claude_recod_img_{args.split}.json'
    print(os.path.join(args.save_path, args.save_name))

    os.makedirs(args.save_path, exist_ok=True)
    os.system("cp claude_recod_img.py %s/" % args.save_path)
    main(args)
