# torchrun benchmark_qa.py
from transformers import AutoTokenizer, LlamaForCausalLM, MistralForCausalLM
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
chencherry = SmoothingFunction()
import time
import os
import argparse
import json
import re
import string
import copy
import nltk.data

import torch
from tqdm import tqdm

import pdb
import numpy as np
import collections
from itertools import chain
import inspect
from rouge_score import rouge_scorer

B_INST, E_INST = "[INST]", "[/INST]"

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--num_steps", type=int, default=5)
    args, _ = args.parse_known_args()
    return args

def normalize_question(question):
    if not question.endswith("?"):
        question = question + "?"

    return question[0].lower() + question[1:]


def build_qa_prompt(example):
    q_prompt = '''\n\nTranslate the above text from English to French. You must only provide the translated text. Do not include any other words or instructions. You must NOT provide any additional information.\n''' 
    
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    fp = open(example)
    data = fp.read()
    doc_prompts = tokenizer.tokenize(data)
    return doc_prompts, q_prompt

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def text_has_answer(answers, text) -> bool:
    if isinstance(answers, str):
        answers = [answers]
    text = normalize_answer(text)
    for single_answer in answers:
        single_answer = normalize_answer(single_answer)
        if single_answer in text:
            return True
    return False


def exact_match(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def substring_match(prediction, ground_truth):
    return normalize_answer(ground_truth) in normalize_answer(prediction)

def get_answer_from_model_output(outputs, prompt):
    answer = outputs.split("\n")[0]
    return answer, None

def normalize_text(s):
    """Removing articles and punctuation, and standardizing 
    whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    
    # if either the prediction or the truth is no-answer 
    # then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    
    return 2 * (prec * rec) / (prec + rec)

def compute_accuracy(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    
    # if either the prediction or the truth is no-answer 
    # then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    return int(set(truth_tokens).issubset(set(pred_tokens)))

def split_ratio(low, num_doc, ratio):
    first_doc_k_ratio_list = []
    increment = (2*ratio-low)/num_doc
    for i in range(num_doc):
        temp_ratio = min(low + i* increment, 1.0)
        first_doc_k_ratio_list.append(temp_ratio)
    return first_doc_k_ratio_list

def split_ratio_uniform(ratio, num_doc):
    return [ratio]*num_doc

def rechunk(ids, chunk_len=100):
    chunk_ids = []
    end_len = chunk_len
    while True:
        if end_len > len(ids):
            if len(ids)-(end_len-chunk_len) < 0.3*chunk_len:
                chunk_ids[-1].extend(ids[end_len-chunk_len:])
            else:
                chunk_ids.append(ids[end_len-chunk_len:])
            break
        else:
            chunk_ids.append(ids[end_len-chunk_len:end_len])
            end_len += chunk_len
    return chunk_ids

def evaluate_dataset(
        model, tokenizer, eval_dataset, index, model_name, num_docs=0, output_dir=None, num_steps=1,
):
    num_layer = 32
    
    check_layers_list = [[0]]
    first_doc_k_ratios_list = [[0.0]]#[[0.02],[0.05],[0.1],[0.15]]
    crash_indices = [97,107]
    
    for first_doc_k_ratios in first_doc_k_ratios_list:
        for check_layers in check_layers_list:
                
            idx_i = 0
            idx = 0
            num_correct_check = 0
            total_len = 0
            total_time = 0
            total_res_toks_len = 0
            inputs_prefill_times = []
            decode_times = []
            total_res_toks = []
            decoding_speeds = []
            for ex in (tq := tqdm(eval_dataset, desc=f"top_k_ratio: 0, EM:  0.0%")):
                doc_prompts, q_prompt = build_qa_prompt(ex)

                print(len(doc_prompts))
                print(num_steps)
                partition_size = len(doc_prompts) // num_steps
                partitions = []
                for i in range(num_steps):
                    if i == num_steps - 1:
                        partitions.append(' '.join(doc_prompts[i * partition_size:]))
                    else:
                        partitions.append(' '.join(doc_prompts[i * partition_size:(i + 1) * partition_size]))
                doc_prompts = partitions
                
                requests = [ChatCompletionRequest(messages=[UserMessage(content=f"'{doc}'")]) for doc in doc_prompts]
                doc_chunk_ids = [tokenizer.encode_chat_completion(request).tokens for request in requests]
                del requests
                
                # doc_chunk_ids = [tokenizer.encode(f"'{doc}'")[1:] for doc in doc_prompts]
                
                doc_chunk_ids_length = [len(ids) for ids in doc_chunk_ids]
                
                print(f"DOC CHUNK IDS LEN: {doc_chunk_ids_length}")
                
                if len(doc_chunk_ids)==0:
                    continue
                
                idx_i += 1
                
                request = ChatCompletionRequest(messages=[UserMessage(content=q_prompt)])
                q_ids = tokenizer.encode_chat_completion(request).tokens#[4:-4]
                del request
                     
                # q_ids = tokenizer.encode(q_prompt)[1:]
                q_ids_len = len(q_ids)
                total_len += q_ids_len
                
                temp_output_past_key_values = []
                #Timer for e2e delay
                e2e_start = time.time()                
                for n in range(num_steps):
                    temp_doc_chunk_ids = doc_chunk_ids[:n+1]
                    input_ids =  list(chain.from_iterable(temp_doc_chunk_ids)) + q_ids + total_res_toks
                    input_tensor = torch.tensor(input_ids, dtype=torch.long).view(1,-1).to(model.device)
                    input_len = len(input_ids)
                    assert input_len == q_ids_len + sum(doc_chunk_ids_length[:n+1]) + len(total_res_toks)
                    total_len += doc_chunk_ids_length[n]
                    
                    chunk_past_key_values = [] #layer k
                    
                    print(f"LEN q_ids: {q_ids_len}")
                    print(f"LEN doc_chunk_ids: {doc_chunk_ids_length[:n+1]}")
                    print(f"LEN input_ids: {input_len}")
                    
                    # kv for prompt and doc
                    with torch.no_grad():
                        # Timer for prefill
                        #warmup
                        output_dict = model(torch.tensor([1], dtype=torch.long).view(1,-1).to(model.device))
                        prefill_start = torch.cuda.Event(enable_timing=True)
                        prefill_end = torch.cuda.Event(enable_timing=True)
                        prefill_start.record()
                        
                        output_dict = model(input_tensor, shift = 0)
                        prefill_end.record()
                        torch.cuda.synchronize()
                        prefill_time = prefill_start.elapsed_time(prefill_end)
                        total_time += prefill_time
                        inputs_prefill_times.append(prefill_time)
                        
                        unprocessed_tok = torch.argmax(output_dict['logits'][:,-1,:])
                        past_key_values = output_dict['past_key_values']
                        assert past_key_values[0][0].shape[2] == input_len
                        for j in range(num_layer):
                            temp_k = past_key_values[j][0][:,:,:input_len].clone() # do not chage with s_start_1
                            temp_v = past_key_values[j][1][:,:,:input_len].clone()
            
                            chunk_past_key_values.append([temp_k, temp_v])
                            
                    chunk_past_key_values = [tuple(x) for x in chunk_past_key_values]
                    chunk_past_key_values = tuple(chunk_past_key_values)

                    
                    start_real_decode=False 
                    
                    temp_input_tensor = input_tensor.clone()                   
                    
                    print(f"LEN input_tensor: {len(input_tensor[0])}, pkv_shape: {chunk_past_key_values[0][0].shape[2]}")
                    
                    
                    imp_indices = [i for i in range(len(input_tensor[0]))]

                    max_gen_len = round(doc_chunk_ids_length[n]*1.8)
                    with torch.no_grad():
                        print(f"INPUTS TO MODEL: {tokenizer.decode(input_tensor.tolist()[0])}")
                        res_toks = []
                        for i in range(max_gen_len):
                            # if input_len+i >= 4096:
                            #     break
                            if i == 0:
                                #warmup
                                output_dict = model(torch.tensor([1], dtype=torch.long).view(1,-1).to(model.device))
                                decode_start = torch.cuda.Event(enable_timing=True)
                                decode_end = torch.cuda.Event(enable_timing=True)
                                decode_start.record()
                                
                                output_dict = model(
                                    input_tensor,
                                    past_key_values=chunk_past_key_values,
                                    check=True,
                                    imp_indices=imp_indices, 
                                    top_k_ratios = first_doc_k_ratios,
                                    check_layers=check_layers,
                                    last_len = 1
                                    )
                            else:
                                output_dict = model(input_tensor, past_key_values=past_key_values)
                            # if i==0:
                            #     torch.cuda.synchronize()
                            #     temp_time = start.elapsed_time(end)
                            #     print(f"org temp time: {temp_time}; input_len: {input_len}")
                            #     total_time+=temp_time
                            tok = torch.argmax(output_dict['logits'][:,-1,:])
                            past_key_values = output_dict['past_key_values']
                            if start_real_decode==False and int(tok) not in [13,29871]:
                                start_real_decode=True
                            if int(tok) == tokenizer.instruct_tokenizer.tokenizer.eos_id:
                                assert len(res_toks) == (past_key_values[0][0].shape[2] - len(temp_input_tensor[0]))
                                break
                            # if int(tok) == 13 and start_real_decode==True:
                            #     break
                            res_toks.append(int(tok))
                            total_res_toks.append(int(tok))
                            input_tensor = tok.view(1,-1)
                        decode_end.record()
                        torch.cuda.synchronize()
                        decode_time = decode_start.elapsed_time(decode_end)
                        total_time += decode_time
                        decode_times.append(decode_time)
                        decoding_speeds.append(len(res_toks)/decode_time)
                        print(f"res_toks_len: {len(res_toks)}, max_gen_len: {max_gen_len}\n")
                        print(f"RES: {tokenizer.decode(res_toks)}")
                        
                        
                    total_len += len(res_toks)
                    total_res_toks_len = len(total_res_toks)
                    del past_key_values                
                    del chunk_past_key_values
                
                
                # with open(f'/dataheart/zhuohan/proactive/transformers_fuse/datasets/en-zh-short.zh/{index}.txt', 'r') as f:
                with open(f'/dataheart/zhuohan/rag-benchmark/proactive_map_reduce/news-commentary-en-fr/en-fr-short.fr/{index}.txt', 'r') as f:
                    gt_text = f.read()
                outputs = tokenizer.decode(total_res_toks)
                reference = [gt_text.split()]
                candidate = outputs.split()
                bleu_score = sentence_bleu(reference, candidate, smoothing_function=chencherry.method7)
                f1 = compute_f1(outputs, gt_text)
                
                
                if output_dir is not None:
                    assert total_time == sum(inputs_prefill_times) + sum(decode_times)
                    d = {
                        "index": index,
                        "doc_chunk_ids_length": doc_chunk_ids_length,
                        "inputs_prefill_times": inputs_prefill_times,
                        "decode_times": decode_times,
                        "e2e_delay": total_time,
                        "decoding_speeds": decoding_speeds,
                        "dataset": "opus",
                        "model": model_name.split('/')[1],
                        "bleu": bleu_score,
                        "f1": f1,
                        "outputs": outputs,
                        }
                    
                    with open(f"{output_dir}proactive2_{num_steps}_steps_{model_name.split('/')[1]}.json", "a") as f:
                        f.write(json.dumps(d) + "\n")         
                                            
            

def load_dataset(dataset_path):
    print("Loading dataset:", dataset_path)
    with open(dataset_path) as f:
        return json.load(f)


def main():
    args = get_args()
    
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    #device = "cpu"
    # model_name = 'mistralai/Mistral-7B-Instruct-v0.2'
    model_name = 'mistralai/Mistral-7B-Instruct-v0.3'
    # model_name = 'meta-llama/Llama-2-70b-chat-hf'
    # model_name = 'meta-llama/Llama-2-7b-chat-hf'
    # model_name = 'meta-llama/Llama-2-13b-chat-hf'
    with torch.no_grad():
        # model = MistralForCausalLM.from_pretrained(model_name,
        #                                     load_in_8bit=True, 
        #                                     use_cache=True,
        #                                     device_map="auto",
        #                                     )
        
        model = LlamaForCausalLM.from_pretrained(model_name,
                                            load_in_8bit=True,
                                            device_map='auto',
                                            #torch_dtype=torch.float16,
                                            #bnb_4bit_compute_dtype=torch.float16,
                                            use_cache=True)#.to(device) #Don't use to for load_in_8/4bit since the model has already been set to the correct devices and casted to the correct `dtype`.
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = MistralTokenizer.v3()

    txt_file = '/dataheart/zhuohan/rag-benchmark/proactive_map_reduce/news-commentary-en-fr/en-fr-short.en/{index}.txt'
    # txt_file = '/dataheart/zhuohan/proactive/transformers_fuse/datasets/en-zh-short.en/{index}.txt'
    # output_dir = "opus_short_outputs_en_fr/"
    output_dir = "opus_short_outputs_en_fr/"

       
    for index in range(5):
        evaluate_dataset(
            model, tokenizer,
            [txt_file.format(index=index)],
            index,
            model_name,
            num_docs=index,
            output_dir=output_dir,
            num_steps = args.num_steps,
        )


if __name__ == '__main__':
    main()