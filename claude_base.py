import os
import random
import argparse
import json
from tqdm import tqdm
import template
import claude


def save_config(args: argparse.Namespace):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.save_path, "args.json")
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile)


def main(args):
    ids = json.load(open(f"data/OK-VQA/OpenEnded_mscoco_{args.split}2014_questions.json"))['questions'] if args.dataset == 'okvqa' \
        else json.load(open(f"data/A-OKVQA/aokvqa_v1p0_{args.split}.json"))
    image_ids = list(set([d['image_id'] for d in ids]))

    dataset = []
    save_config(args)
    pbar = tqdm(total=len(image_ids))
    for idx, target_idx in enumerate(image_ids):
        target_path = os.path.join(args.data_dir, f"{args.split}2014/COCO_{args.split}2014_{target_idx:012d}.jpg") if args.dataset == 'okvqa' \
            else os.path.join(args.data_dir, f"{args.split}2017/{target_idx:012d}.jpg")

        datum = {'image_id': target_idx, 'new_captions': []}

        for loop in range(args.loop):
            args.config['temperature'] = random.random()
            response = claude.one_image(target_path, "image/jpeg", args.query, args.config)
            differences = response.content[0].text.strip()
            datum['new_captions'].append(differences)

        dataset.append(datum)
        pbar.update(1)

    with open(os.path.join(args.save_path, args.save_name), 'w') as outfile:
        json.dump(dataset, outfile, indent=4)

    gen_infer_file(os.path.join(args.save_path, args.save_name), args)


def gen_infer_file(raw_file, args):
    if args.dataset == 'okvqa':
        question = json.load(open(f"data/OK-VQA/OpenEnded_mscoco_{args.split}2014_questions.json"))['questions']
        answer = json.load(open(f"data/OK-VQA/mscoco_{args.split}2014_annotations.json"))['annotations']
        caption = json.load(open(raw_file))

        loop_dict = [{} for _ in range(0, args.loop)]
        for item in caption:
            for idx, l in enumerate(range(0, args.loop)):
                loop_dict[idx][item['image_id']] = item['new_captions'][:l+1]

        for idx, l in enumerate(range(0, args.loop)):
            for q, a in zip(question, answer):
                q['answers'] = a['answers']
                loop_can = loop_dict[idx][q['image_id']]
                q['new_captions'] = loop_can
            if args.split == 'train': json.dump(question, open(f"diff/okvqa/claude_base_train/claude_train_final.json", "w"), indent=4)
            else:                     json.dump(question, open(f"diff/okvqa/claude_base_{args.split}/claude_base_{args.split}_total{l+1}.json", "w"), indent=4)
    else:
        aokvqa = json.load(open(f"data/A-OKVQA/aokvqa_v1p0_{args.split}.json"))
        caption = json.load(open(raw_file))

        loop_dict = [{} for _ in range(0, args.loop)]
        for item in caption:
            for idx, l in enumerate(range(0, args.loop)):
                loop_dict[idx][item['image_id']] = item['new_captions'][:l+1]

        for idx, l in enumerate(range(0, args.loop)):
            for d in aokvqa:
                loop_can = loop_dict[idx][d['image_id']]
                d['new_captions'] = loop_can
            if args.split == 'train': json.dump(aokvqa, open(f"diff/aokvqa/claude_base_train/claude_train_final.json", "w"), indent=4)
            else:                     json.dump(aokvqa, open(f"diff/aokvqa/claude_base_{args.split}/claude_base_{args.split}_total{l+1}.json", "w"), indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument("--data_dir", type=str, default="data/COCO/")
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
    args.query = template.claude

    if args.split == 'train':
        args.loop = 1
    args.save_path = 'diff/{}/claude_base_{}'.format(args.dataset, args.split)
    args.save_name = f'claude_base_{args.split}.json'
    print(os.path.join(args.save_path, args.save_name))

    os.makedirs(args.save_path, exist_ok=True)
    os.system("cp claude_base.py %s/" % args.save_path)
    main(args)
