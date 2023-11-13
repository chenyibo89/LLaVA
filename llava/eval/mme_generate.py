import argparse
import torch
import os

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    bf16_flag = False
    print(model_name.lower())
    if 'sam' in model_name.lower():
        if 'avg' in model_name.lower():
            print('load sam-avg model')
            from llava.model_sam_pool.builder import load_pretrained_model
        elif 'clip' in model_name.lower():
            print('load clip-sam model')
            from llava.model_clip_sam.builder import load_pretrained_model
        else:
            print('load sam-ca model')
            from llava.model_sam.builder import load_pretrained_model
            bf16_flag = True
    elif 'CA' in model_name:
        if 'mlp' in  model_name:
           print('load CA-mlp model')
           from llava.model_ca_mlp.builder import load_pretrained_model
        else:
            print('load CA model')
            from llava.model_ca.builder import load_pretrained_model
    elif 'PR' in model_name:
        print('load PR model')
        from llava.model_pr.builder import load_pretrained_model
    else:
        print('load llava model')
        from llava.model.builder import load_pretrained_model
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    if bf16_flag:
        for name, module in model.named_modules():
            module = module.to(torch.bfloat16)
    
    eval_type_dict = {
        "Perception": ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"],
        "Cognition": ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]
    }

    type_1_task = ['artwork', 'celebrity', 'landmark', 'posters', 'scene']
    type_2_task = ['code_reasoning', 'color', 'commonsense_reasoning', 'count', 'existence', 'numerical_calculation', 'OCR', 'position', 'text_translation']

    for eval_type, task_name_list in eval_type_dict.items():
        print("===========", eval_type, "===========")

        for task_name in task_name_list:
            base_dir = os.path.join(args.base_dir, task_name)
            f = open(os.path.join(args.results_dir, task_name + '.txt'), 'w')
            
            if task_name in type_1_task:
                image_dir = os.path.join(base_dir, 'images')
                if args.mode == 'en':
                    question_answers_dir = os.path.join(base_dir, 'questions_answers_YN')
                else:
                    question_answers_dir = os.path.join(base_dir, 'questions_answers_YN_zh')
            else:
                question_answers_dir = base_dir
                image_dir = base_dir

            for image_filename in os.listdir(image_dir):
                if image_filename[-4:] == '.txt':
                    continue
                # print(image_filename)
                if args.mode == 'zh' and task_name in type_2_task:
                    question_answer_filename = os.path.join(question_answers_dir, image_filename[:-4] + '_zh.txt')
                else:
                    question_answer_filename = os.path.join(question_answers_dir, image_filename[:-4] + '.txt')
                lines = open(question_answer_filename, 'r').readlines()
                # one image corresponds to two questions
                for index in range(2):
                    question, groundtruth = lines[index].split("\t")
                    image_path = os.path.join(image_dir, image_filename)
                    qs = question
                    cur_prompt = qs
                    if model.config.mm_use_im_start_end:
                        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
                    else:
                        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
                    conv = conv_templates[args.conv_mode].copy()
                    conv.append_message(conv.roles[0], qs)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()

                    if args.conv_mode == "lingji_2":
                        prompt += "<duer-assistant>"
                        print("conversation: {}".format(prompt))
                        from llava.mm_utils import tokenizer_image_token_for_lingji
                        input_ids = tokenizer_image_token_for_lingji(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                        # print("input ids: {}".format(input_ids.tolist()))
                    else:
                        print("conversation: {}".format(prompt))
                        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

                    image = Image.open(image_path).convert('RGB')
                    if 'clip-sam' in model_name.lower():
                        images = [image]
                    else:
                        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                    # print(image_processor)

                    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    keywords = [stop_str]
                    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                    if 'clip-sam' not in model_name.lower():
                        if bf16_flag:
                            images = image_tensor.unsqueeze(0).to(torch.bfloat16).cuda()
                        else:
                            images = image_tensor.unsqueeze(0).half().cuda()
                    # print(input_ids)

                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids,
                            images=images,
                            do_sample=True,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            num_beams=args.num_beams,
                            # no_repeat_ngram_size=3,
                            max_new_tokens=1024,
                            use_cache=True)

                    input_token_len = input_ids.shape[1]
                    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                    if n_diff_input_output > 0:
                        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
                    # print('output:', outputs)
                    outputs = outputs.strip()
                    if outputs.endswith(stop_str):
                        outputs = outputs[:-len(stop_str)]
                    outputs = outputs.strip()
                    answer = outputs.replace('\n', '')
                    f.write(image_filename + '\t' + lines[index].replace('\n', '') + '\t' + answer + '\n')

            f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="llava")
    parser.add_argument("--model-path", type=str, default="/pfs-LLM/public/infra/chenyibo02/code/LLaVA/checkpoints/llava-llama-2-13b-chat-sft-full-1ep")
    parser.add_argument('--results_dir', default='/pfs-LLM/public/infra/wuhao/code/MME/Model_Results/llava-llama-2-13b-chat-sft-full-1ep/', type=str)
    parser.add_argument('--base_dir', default="/pfs-LLM/public/infra/wuhao/code/MME/MME_Benchmark_release/", type=str)
    parser.add_argument('--mode', default='en')
    
    parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--conv-mode", type=str, default="conv_llava_llama_2")
    parser.add_argument("--conv-mode", type=str, default="llava_llama_2")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)

    args = parser.parse_args()
    args.results_dir = os.path.join(args.results_dir, args.mode)
    os.makedirs(args.results_dir, exist_ok=True)
    # os.mkdir(args.results_dir)
    # os.mkdir(args.results_dir + args.mode)
    eval_model(args)
