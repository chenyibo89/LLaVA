import argparse
import torch
import os

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
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
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # 子任务列表
    eval_type_dict = {
        "Perception": ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"],
        "Cognition": ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]
    }

    # 任务共分成两类，这两类的区别主要是他们在MME_Benchmark_release中的目录结构不同
    # type1中的任务的目录结构中，图片存放在子任务的images文件夹下面，对应的问题和答案在子任务的questions_answers_YN(_zh)文件夹下面
    # type2中的任务的目录结构中，图片、对应的问题和答案都混在子任务目录下面
    # 因此下文的处理需要分别去处理type1和type2的任务
    type_1_task = ['artwork', 'celebrity', 'landmark', 'posters', 'scene']
    type_2_task = ['code_reasoning', 'color', 'commonsense_reasoning', 'count', 'existence', 'numerical_calculation', 'OCR', 'position', 'text_translation']

    for eval_type, task_name_list in eval_type_dict.items():
        print("===========", eval_type, "===========")

        for task_name in task_name_list:
            base_dir = os.path.join(args.base_dir, task_name)
            f = open(args.results_dir + task_name + '.txt', 'w')
            
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
                    question_answer_filename = question_answers_dir + image_filename[:-4] + '_zh.txt'
                else:
                    question_answer_filename = question_answers_dir + image_filename[:-4] + '.txt'
                lines = open(question_answer_filename, 'r').readlines()
                # one image corresponds to two questions
                for index in range(2):
                    # print(lines[index])
                    question, groundtruth = lines[index].split("\t")
                    image_path = image_dir + image_filename
                    qs = question
                    # qs = '图片中有什么？请详细描述一下图片中的内容'
                    # qs = 'What\'s in the image? Please describe in detail.'
                    cur_prompt = qs
                    if model.config.mm_use_im_start_end:
                        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
                    else:
                        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
                    conv = conv_templates[args.conv_mode].copy()
                    conv.append_message(conv.roles[0], qs)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()

                    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

                    image = Image.open(image_path)
                    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

                    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    keywords = [stop_str]
                    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                    with torch.inference_mode():
                        outputs = model.generate(
                            input_ids,
                            images=image_tensor.unsqueeze(0).half().cuda(),
                            do_sample=False,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            num_beams=args.num_beams,
                            # no_repeat_ngram_size=3,
                            max_new_tokens=1024,
                            use_cache=True,
                            return_dict_in_generate=True,
                            output_scores=True
                            )
                        # print(outputs.scores)
                    output_ids = outputs.sequences
                    input_token_len = input_ids.shape[1]
                    # let's stack the logits generated at each step to a tensor and transform logits to probs
                    max_token_probability = -1
                    
                    ind = 1                 # TODO: Automatically change this index
                    probs = torch.stack(outputs.scores, dim=1).softmax(-1)  # -> shape [1, seq_len, vocab_size]
                    max_id = torch.argmax(probs[0][ind]).item()

                    a, idx1 = torch.sort(probs[0][ind], descending=True)#descending为alse，升序，为True，降序
                    idx = idx1[:3]
                    outputxx = tokenizer.batch_decode(idx, skip_special_tokens=True)
                    max_token_probability = [probs[0][ind][idx[i]].item() for i in range(3)]
                    print(outputxx)
                    print(max_token_probability)

                    # outputxx = tokenizer.batch_decode([max_id], skip_special_tokens=True)[0]
                    # if outputxx == '是' or outputxx == '否' or outputxx == '不' or outputxx == '对' \
                    #     or outputxx == 'Yes' or outputxx == 'No' or outputxx == 'yes' or outputxx == 'no':
                    #     print(str(max_id) + '\t' + outputxx + '\t' + str(max_token_probability))
                    #     break

                    input_token_len = input_ids.shape[1]
                    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                    if n_diff_input_output > 0:
                        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
                    outputs = outputs.strip()
                    if outputs.endswith(stop_str):
                        outputs = outputs[:-len(stop_str)]
                    outputs = outputs.strip()
                    answer = outputs.replace('\n', '')
                    # print(lines[index])
                    # 除了正常的输出之外，还要把排行前三的token是什么以及对应的概率写入文件中
                    f.write(image_filename + '\t' + lines[index].replace('\n', '') + '\t' + answer + '\t' 
                            + outputxx[0] + '\t' + str(max_token_probability[0]) + '\t'
                            + outputxx[1] + '\t' + str(max_token_probability[1]) + '\t'
                            + outputxx[2] + '\t' + str(max_token_probability[2]) + '\n')

            f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="llava")
    parser.add_argument("--model-path", type=str, default="/home/users/wuhao33/work/model/llava-chinese-alpaca-2-7b-sft-chnVIT-chnData-full-1ep/")
    # parser.add_argument("--model-path", type=str, default="/home/users/lilinyu03/work/llava-llama-2-13b-chat-lightning-preview/")
    parser.add_argument('--results_dir', default='/home/users/lilinyu03/work/MME/Model_Results/llava-chinese-alpaca-2-7b-sft-chnVIT-chnData-full-1ep/cognition_revised/', type=str)
    parser.add_argument('--base_dir', default="/home/users/lilinyu03/work/MME/MME_Benchmark_release/", type=str)
    parser.add_argument('--mode', default='zh')
    
    parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--conv-mode", type=str, default="conv_llava_llama_2")
    parser.add_argument("--conv-mode", type=str, default="llava_llama_2")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)

    args = parser.parse_args()
    args.results_dir = args.results_dir + args.mode + '/'
    os.makedirs(args.results_dir, exist_ok=True)
    # os.mkdir(args.results_dir)
    # os.mkdir(args.results_dir + args.mode)
    eval_model(args)