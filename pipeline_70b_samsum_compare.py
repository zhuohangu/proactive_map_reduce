# torchrun benchmark_qa.py
from transformers import AutoTokenizer, LlamaForCausalLM
import os
import argparse
import json
import re
import string
import copy
import time

import torch
from tqdm import tqdm

import pdb
import numpy as np
import collections
from itertools import chain
import inspect
from rouge_score import rouge_scorer
from kv_store import add_kv_layer, fetch_kv_layer, fetch_kv
from random import sample

B_INST, E_INST = "[INST]", "[/INST]"

def normalize_question(question):
    if not question.endswith("?"):
        question = question + "?"

    return question[0].lower() + question[1:]


def build_qa_prompt(example):

    q = "\n\n"+example["question"]
    doc_prompts = [f"{ctx['text']}" for ctx in example["ctxs"]]
    #ex_prompt = f"{docs_text}\n\nBased on these texts, answer the question:\nQ: {q}\nA:"
    q_prompt = f"{q}"
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

def compute_r1(pred, gold):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge1 = scorer.score(gold, pred)['rouge1'].fmeasure
    rougeL = scorer.score(gold, pred)['rougeL'].fmeasure
    return rougeL#rouge1, rougeL

def compute_f1(a_pred, a_gold, tokenizer):
    gold_toks = tokenizer.encode(normalize_answer(a_gold))[1:]
    pred_toks = tokenizer.encode(normalize_answer(a_pred))[1:]
    #pdb.set_trace()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

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
        model, tokenizer, eval_dataset, num_docs=0, output_dir=None
):
    num_layer = 32
    max_gen_len = 128
    

    check_layers_list = [[1]]
    first_doc_k_ratios_list = [[0.15]]#[[0.02],[0.05],[0.1],[0.15]]
    crash_indices = [97,107]
    
    for first_doc_k_ratios in first_doc_k_ratios_list:
        for check_layers in check_layers_list:
                
            idx_i = 0
            idx = 0
            num_correct_check = 0
            total_len = 0
            total_time = 0
            for ex in (tq := tqdm(eval_dataset, desc=f"top_k_ratio: 0, EM:  0.0%")):
                

                doc_prompts, q_prompt = build_qa_prompt(ex)
                doc_chunk_ids = [tokenizer.encode(doc)[1:] for doc in doc_prompts]
                
                #pre-chunk d drop extras
                #doc_chunk_ids = rechunk(list(chain.from_iterable(doc_chunk_ids)), chunk_len=100)
                
                while len(list(chain.from_iterable(doc_chunk_ids))) > 3400:
                    del_idx = int(len(doc_chunk_ids)/2)
                    del doc_chunk_ids[del_idx]
                
                #doc_chunk_ids = rechunk(list(chain.from_iterable(doc_chunk_ids)), chunk_len=chunk_len)
                
                if len(doc_chunk_ids)==0:
                    continue
                
                #if idx_i not in crash_indices:
                #    idx_i+=1
                #    print("skipped due to crash")
                #    continue
                idx_i += 1
                     
                q_ids = tokenizer.encode(q_prompt)[1:]

                prefix_prompt = "Summarize the dialogue into a few short sentences. The following are some examples.\n\n"
                s_start = [1] + tokenizer.encode(prefix_prompt)[1:]
                s_start_len = len(s_start)
                s_start_1 = [1]
                s_start_1_len = 1
                s_end = []
                s_end_len = len(s_end)

                input_ids = s_start + list(chain.from_iterable(doc_chunk_ids)) + q_ids
                input_tensor = torch.tensor(input_ids, dtype=torch.long).view(1,-1).to(model.device)
                input_len = len(input_ids)
                total_len += input_len

                
                test_last_len = len(q_ids+s_end)

                
                #pdb.set_trace()

                input_chunk_tensors = []
                temp_input = s_start
                input_chunk_tensors.append(torch.tensor(temp_input, dtype=torch.long).view(1,-1).to(model.device))

                for i in range(len(doc_chunk_ids)):
                    temp_input = doc_chunk_ids[i]
                    if i == 0:
                        input_chunk_tensors.append(torch.tensor(s_start_1 + temp_input, dtype=torch.long).view(1,-1).to(model.device))    
                    else:
                        input_chunk_tensors.append(torch.tensor(s_start_1 + temp_input, dtype=torch.long).view(1,-1).to(model.device))                
                
                input_chunk_tensors.append(torch.tensor(s_start_1 + q_ids + s_end, dtype=torch.long).view(1,-1).to(model.device))

                shifts = []
                imp_indices = []
                cur_pos = 0
            
                for i in range(len(input_chunk_tensors)):
                    
                    doc_len = input_chunk_tensors[i].shape[1]-s_start_1_len
                    
                    shifts.append(cur_pos)
                    cur_pos += doc_len
                
                

                chunk_past_key_values = [] #layer k
                
                #Generate and concatenate kv for chunks
                with torch.no_grad():
                    for i in range(len(input_chunk_tensors)):
                        output_dict = model(input_chunk_tensors[i], shift=shifts[i])
                        past_key_values = output_dict['past_key_values']
                        for j in range(num_layer):
                            if i == 0:
                                temp_k = past_key_values[j][0][:,:,:s_start_len].clone() # do not chage with s_start_1
                                temp_v = past_key_values[j][1][:,:,:s_start_len].clone()
                            elif i == 1:
                                temp_k = past_key_values[j][0][:,:,s_start_1_len:s_start_1_len+len(doc_chunk_ids[i-1])].clone()
                                temp_v = past_key_values[j][1][:,:,s_start_1_len:s_start_1_len+len(doc_chunk_ids[i-1])].clone()    
                            elif i == len(input_chunk_tensors)-1:
                                temp_k = past_key_values[j][0][:,:,s_start_1_len:s_start_1_len+len(q_ids)+s_end_len].clone()
                                temp_v = past_key_values[j][1][:,:,s_start_1_len:s_start_1_len+len(q_ids)+s_end_len].clone()
                            else:
                                temp_k = past_key_values[j][0][:,:,s_start_1_len:s_start_1_len+len(doc_chunk_ids[i-1])].clone()
                                temp_v = past_key_values[j][1][:,:,s_start_1_len:s_start_1_len+len(doc_chunk_ids[i-1])].clone()

                            if i == 0:
                                chunk_past_key_values.append([temp_k, temp_v])

                            else:
                                chunk_past_key_values[j][0] = torch.cat((chunk_past_key_values[j][0],temp_k), dim=2)
                                chunk_past_key_values[j][1] = torch.cat((chunk_past_key_values[j][1],temp_v), dim=2)
                        #pdb.set_trace()
                
                
                #for j in range(num_layer):
                #    add_kv_layer(chunk_past_key_values[j][0], "key", j, tier="cpu_pin")
                #    add_kv_layer(chunk_past_key_values[j][1], "value", j, tier="cpu_pin")     
                
                imp_indices = [i for i in range(len(input_ids))]
                
                kv_dtype = chunk_past_key_values[j][0].dtype
                temp_device = chunk_past_key_values[j][0].device
                kv_shape_org = chunk_past_key_values[j][0].shape
                cpu_k_org = torch.rand(kv_shape_org, dtype=kv_dtype)#.pin_memory()
                cpu_v_org = torch.rand(kv_shape_org, dtype=kv_dtype)#.pin_memory()
                temp_k = torch.empty(kv_shape_org, dtype=kv_dtype, device=temp_device)
                temp_v = torch.empty(kv_shape_org, dtype=kv_dtype, device=temp_device)
                test_metadata = {"kv_dtype": kv_dtype,
                                 "temp_device": temp_device,
                                 "kv_shape_org":kv_shape_org,
                                 "cpu_k_org":cpu_k_org,
                                 "cpu_v_org":cpu_v_org,
                                 "temp_k": temp_k,
                                 "temp_v":temp_v}
                #pdb.set_trace()
                load_indices = imp_indices.copy()
                time.sleep(2)
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for j in range(num_layer-1):
                    #cpu_k = cpu_k_org[:,:,load_indices]
                    #cpu_v = cpu_v_org[:,:,load_indices]
                    #temp_k[:,:,load_indices].copy_(cpu_k)
                    #temp_v[:,:,load_indices].copy_(cpu_v)
                    
                    #cpu_k = cpu_k_org
                    #cpu_v = cpu_v_org
                    #temp_k.copy_(cpu_k)
                    #temp_v.copy_(cpu_v)
                    
                    cpu_k = cpu_k_org.pin_memory()
                    cpu_v = cpu_v_org.pin_memory()
                    temp_k.copy_(cpu_k)
                    temp_v.copy_(cpu_v)
                    #pdb.set_trace()
                    #temp_k.copy_(cpu_k_org)
                    #temp_v.copy_(cpu_v_org)
                    #if j > 0:
                    #    load_indices = sample(imp_indices, int(input_len*(1-first_doc_k_ratios[0])))
                end.record()
                torch.cuda.synchronize()
                temp_time = start.elapsed_time(end)
                print(f"load time: {temp_time}")
                
                # construct imp_indices
                
                chunk_past_key_values = [tuple(x) for x in chunk_past_key_values]
                chunk_past_key_values = tuple(chunk_past_key_values)
                del past_key_values
                
                input_tensor = torch.tensor(input_ids, dtype=torch.long).view(1,-1).to(model.device)
                    

                
                input_tensor = torch.tensor(input_ids, dtype=torch.long).view(1,-1).to(model.device)
                with torch.no_grad():
                    #warmup
                    output_dict = model(torch.tensor([1], dtype=torch.long).view(1,-1).to(model.device))
                    
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    #FIXME(Jiayi): Don't use this many params, use a metadat_cls
                    print("-------------------------")
                    output_dict = model(input_tensor, 
                                    past_key_values=chunk_past_key_values,
                                    check=True,
                                    imp_indices=imp_indices, 
                                    top_k_ratios = first_doc_k_ratios,
                                    check_layers=check_layers,
                                    last_len = test_last_len)
                    
                    end.record()
                    torch.cuda.synchronize()
                    temp_time = start.elapsed_time(end)
                    print(f"check (gpu) temp time: {temp_time}; input_len: {input_len}")
                pdb.set_trace()
                input_tensor = torch.tensor(input_ids, dtype=torch.long).view(1,-1).to(model.device)
                with torch.no_grad():
                    #warmup
                    output_dict = model(torch.tensor([1], dtype=torch.long).view(1,-1).to(model.device))
                    print("-------------------------")
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    #FIXME(Jiayi): Don't use this many params, use a metadat_cls
                    output_dict = model(input_tensor)
                    
                    end.record()
                    torch.cuda.synchronize()
                    temp_time = start.elapsed_time(end)
                    print(f"org temp time: {temp_time}; input_len: {input_len}")
                    
                del output_dict
                pdb.set_trace()
                

                idx += 1


                print(idx)
                
                tq.set_description(f"first doc k ratio:{first_doc_k_ratios};")
                
                if idx == 10:
                    pdb.set_trace()
                
                #if idx==30:
                    #pdb.set_trace()
                #    break
            
            

            em_check = num_correct_check / idx
            avg_len = total_len/idx
            avg_time = total_time / idx
            
            
            if output_dir is not None:
                d = {"first_doc_k_ratio":first_doc_k_ratios,
                    #"sample": "30",
                    "r1_check": em_check,
                    "time": avg_time,
                    "avg_len": avg_len,
                    "check_layers": check_layers,
                    "dataset": "samsum",
                    "model": "70b"}

                with open("logs/final_70b.json", "a") as f:
                    f.write(json.dumps(d) + "\n")


def load_dataset(dataset_path):
    print("Loading dataset:", dataset_path)
    with open(dataset_path) as f:
        return json.load(f)


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    #device = "cpu"
    with torch.no_grad():
        model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                            #load_in_8bit=True,
                                            #device_map='auto',
                                            torch_dtype=torch.float16,
                                            #bnb_4bit_compute_dtype=torch.float16,
                                            use_cache=True).to(device) #Don't use to for load_in_8/4bit since the model has already been set to the correct devices and casted to the correct `dtype`.
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    
    eval_dataset = load_dataset("datasets/samsum.json")
    
    for num_doc in [10]:
        evaluate_dataset(
            model, tokenizer,
            eval_dataset,
            num_docs=num_doc,
            output_dir="outputs/",
        )


if __name__ == '__main__':
    main()