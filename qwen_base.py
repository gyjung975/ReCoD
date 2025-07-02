import os
import random
import argparse
import json
import torch
from tqdm import tqdm
from PIL import Image
import template

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


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


@torch.no_grad()
def main(args):
    model = Qwen2VLForConditionalGeneration.from_pretrained(f"Qwen/Qwen2-VL-{args.size}B", torch_dtype="auto", device_map="auto", cache_dir='.cache')
    processor = AutoProcessor.from_pretrained(f"Qwen/Qwen2-VL-{args.size}B-instruct", cache_dir='.cache')

    ids = json.load(open(f"data/OK-VQA/OpenEnded_mscoco_{args.split}2014_questions.json"))['questions'] if args.dataset == 'okvqa' \
        else json.load(open(f"data/A-OKVQA/aokvqa_v1p0_{args.split}.json"))
    image_ids = list(set([d['image_id'] for d in ids]))

    dataset = []
    save_config(args)
    for idx in tqdm(range(0, len(image_ids), args.batch_size)):
        target_ids = image_ids[idx:idx+args.batch_size]
        target_path = [os.path.join(args.data_dir, f"{args.split}2014/COCO_{args.split}2014_{target_id:012d}.jpg") for target_id in target_ids] if args.dataset == 'okvqa' \
            else [os.path.join(args.data_dir, f"{args.split}2017/{target_id:012d}.jpg") for target_id in target_ids]

        datum = [{'image_id': int(target_ids[d])} for d in range(len(target_ids))]

        difference_captions = [[] for _ in target_ids]
        for loop in range(args.loop):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": target_path[0]},
                        {"type": "text", "text": args.query}
                    ]
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                # videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            temperature, top_p = random.random(), random.random()
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            differences = [caption_processing([cap]) for cap in output_text]

            for e in range(len(target_ids)):
                difference_captions[e] += differences[e]

        for d in range(len(datum)):
            datum[d]['new_captions'] = difference_captions[d]
        dataset += datum

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
            if args.split == 'train': json.dump(question, open(f"diff/okvqa/qwen_base_train/qwen_train_final.json", "w"), indent=4)
            else:                     json.dump(question, open(f"diff/okvqa/qwen_base_{args.split}/qwen_base_{args.split}_total{l+1}.json", "w"), indent=4)
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
            if args.split == 'train': json.dump(aokvqa, open(f"diff/aokvqa/qwen_base_train/qwen_train_final.json", "w"), indent=4)
            else: json.dump(aokvqa, open(f"diff/aokvqa/qwen_base_{args.split}/qwen_base_{args.split}_total{l+1}.json", "w"), indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument("--data_dir", type=str, default="data/COCO/")
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
    parser.add_argument("--size", type=int, default=7)
    args = parser.parse_args()

    args.query = template.llava

    if args.split == 'train':
        args.loop = 1
    args.save_path = 'diff/{}/qwen_base_{}'.format(args.dataset, args.split)
    args.save_name = f'qwen_base_{args.split}.json'
    print(os.path.join(args.save_path, args.save_name))

    os.makedirs(args.save_path, exist_ok=True)
    os.system("cp qwen_base.py %s/" % args.save_path)
    main(args)
