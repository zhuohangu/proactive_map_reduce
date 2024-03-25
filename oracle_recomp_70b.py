# torchrun benchmark_qa.py
from transformers import AutoTokenizer, LlamaForCausalLM
import os
import argparse
import json
import re
import string
import copy

import torch
from tqdm import tqdm

import pdb
import numpy as np
import collections
from itertools import chain
import inspect

B_INST, E_INST = "[INST]", "[/INST]"

def normalize_question(question):
    if not question.endswith("?"):
        question = question + "?"

    return question[0].lower() + question[1:]


def build_qa_prompt(example):

    q = normalize_question(example["question"])
    doc_prompts = [f"{ctx['title']}\n\n{ctx['text']}\n\n" for ctx in example["ctxs"]]
    #ex_prompt = f"{docs_text}\n\nBased on these texts, answer the question:\nQ: {q}\nA:"
    q_prompt = f"\n\nAnswer the question based on the given passages. Only give me the exact answer and do NOT output the logic behind the answer or any other words.\n\nQuestion: {q}\nAnswer:"

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
    num_layer = 80
    #num_head = 32
    max_gen_len = 32
    #top_k_ratios = [0.0, 0.05, 0.15]#,0.05,0.1,0.15]
    #last_k_ratios = [0.0, "q_only", 0.05, 0.15]#[0.025]
    #top_k_ratios = [0.0]#,0.05,0.1,0.15]
    last_k_ratios = ["q_only"]#[0.025]#"first_doc_k_ratio",
    #first_doc_k_ratios = [0.0]
    first_doc_k_ratios = [0.0]#0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17]#0.15,0.1,0.0]#[0.15, 0.05, 0.1, 0.45,0.6]
    measure_time = False
    check_layers = [1]
    #top_k_ratio = 0.0
    space_id = 259
    #chunk_len = 100
    
    check_layers = [1]
    hke_ratios = [0.02, 0.05, 0.08]
    first_doc_k_ratio = 0
    
    chunk_lens = [200]
    last_k_ratio_i = "q_only"
    use_layers = [[0]]
    for use_layer in use_layers:
        for hke_ratio in hke_ratios:
            for check_layer in check_layers:
                if hke_ratio==0:
                    check_layer=0
                
                idx_i = 0
                idx = 0
                num_correct_org = 0
                num_correct_check = 0
                num_correct_overlap = 0
                total_len = 0
                real_recomp_ratios = []
                for ex in (tq := tqdm(eval_dataset, desc=f"top_k_ratio: 0, EM:  0.0%")):
                    
                    
                    answers = ex["answers"]

                    doc_prompts, q_prompt = build_qa_prompt(ex)
                    doc_chunk_ids = [tokenizer.encode(doc)[1:] for doc in doc_prompts]
                    
                    #pre-chunk d drop extras
                    #doc_chunk_ids = rechunk(list(chain.from_iterable(doc_chunk_ids)), chunk_len=100)
                    
                    while len(list(chain.from_iterable(doc_chunk_ids))) > 3800:
                        del_idx = int(len(doc_chunk_ids)/2)
                        del doc_chunk_ids[del_idx]
                    
                    #doc_chunk_ids = rechunk(list(chain.from_iterable(doc_chunk_ids)), chunk_len=chunk_len)
                    
                    if len(doc_chunk_ids)==0:
                        continue
                        
                    #Drop last chunk
                    #doc_chunk_ids = rechunk(list(chain.from_iterable(doc_chunk_ids)), chunk_len=chunk_len)[:-2]
                    
                    
                    
                    #if len(doc_chunk_ids) < 2:
                    #    continue
                            
                    q_ids = tokenizer.encode(q_prompt)[1:]

                    prefix_prompt = "You need to answer the question at the end according to the following passages. Here are the passages.\n"
                    s_start = [1, 518, 25580, 29962] + tokenizer.encode(prefix_prompt)[1:]
                    s_start_len = len(s_start)
                    s_start_1 = [1]
                    s_start_1_len = 1
                    s_end = [518, 29914, 25580, 29962]#, 29871, 18585, 29991, 2266, 29915, 29879, 278, 1234, 2729, 373, 278, 2183, 1209, 1179, 29901, 13, 13,]#[518, 29914, 25580, 29962, 29871]
                    s_end_len = len(s_end)

                    input_ids = s_start + list(chain.from_iterable(doc_chunk_ids)) + q_ids + s_end
                    input_tensor = torch.tensor(input_ids, dtype=torch.long).view(1,-1).to(model.device)
                    input_len = len(input_ids)
                    total_len += input_len

                    
                    test_last_len = len(q_ids+s_end)

                    # Normal generation
                    org_post_attn = []
                    with torch.no_grad():
                        res_toks = []
                        for i in range(max_gen_len):
                            if i==0:
                                output_dict = model(input_tensor)
                                org_past_key_values = output_dict['past_key_values']
                                #for j in range(num_layer):
                                    #org_post_attn.append(model.model.layers[j].self_attn.attention_weights_post[0,:,-test_last_len:].clone())
                                    #del model.model.layers[j].self_attn.attention_weights_post
                            else:
                                output_dict = model(input_tensor, past_key_values=past_key_values)
                            tok = torch.argmax(output_dict['logits'][:,-1,:])
                            past_key_values = output_dict['past_key_values']
                            if int(tok) == tokenizer.eos_token_id:
                                break
                            res_toks.append(int(tok))
                            input_tensor = tok.view(1,-1)
                            '''
                            if i == 0:
                                org_output = outputs[0,-1].clone()
                                org_scores_pre = []
                                org_scores_post = []
                                for j in range(num_layer):
                                    org_scores_pre.append(model.layers[j].attention.scores_pre.clone())
                                    org_scores_post.append(model.layers[j].attention.scores_post.clone())
                            '''    
                        res_org = tokenizer.decode(res_toks)
                    
                    
                    #pdb.set_trace()
                    
                    '''
                    org_k = []
                    org_v = []
                    for j in range(num_layer):
                        temp_k = model.layers[j].attention.cache_k[:,:input_len].clone()
                        temp_v = model.layers[j].attention.cache_v[:,:input_len].clone()
                        org_k.append(temp_k)
                        org_v.append(temp_v)
                    '''

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

                    #build shifts and imp_indices
                    shifts = []
                    imp_indices = []
                    cur_pos = 0
                    cur_pos_final = 0
                    
                    if last_k_ratio_i == "q_only":
                        last_k_ratio = (len(q_ids+s_end)+1)/len(input_ids)
                    elif last_k_ratio_i == "last_one":
                        last_k_ratio = 1/len(input_ids)
                    else:
                        #last_k_ratio = last_k_ratio_i 
                        last_k_ratio = "q_only_plus" #HACK
                    
                    
                    first_doc_k_ratio_i = first_doc_k_ratio #(first_doc_k_ratio-last_k_ratio)/(1-last_k_ratio) #HACK
                    if first_doc_k_ratio > 0:
                        #first_doc_k_ratio_list = split_ratio_uniform(first_doc_k_ratio_i, num_doc=len(doc_chunk_ids))
                        first_doc_k_ratio_list = split_ratio(low=first_doc_k_ratio_i/3, num_doc=len(doc_chunk_ids), ratio=first_doc_k_ratio_i)
                        #first_doc_k_ratio_list = split_ratio(low=0.05, num_doc=len(doc_chunk_ids), ratio=first_doc_k_ratio_i)
                    else:
                        first_doc_k_ratio_list = [0.0] * len(doc_chunk_ids)
                        
                    #pdb.set_trace()
                    #first_doc_k_ratio_list = split_ratio_uniform(first_doc_k_ratio, num_doc=num_docs)

                    for i in range(len(input_chunk_tensors)):
                        imp_indices_doc = []
                        #if i==0:# or i==1:
                        #    doc_len = input_chunk_tensors[i].shape[1]-s_start_len
                        #else:
                        #    doc_len = input_chunk_tensors[i].shape[1]-s_start_1_len
                        
                        doc_len = input_chunk_tensors[i].shape[1]-s_start_1_len

                        if i == len(input_chunk_tensors)-1:
                            if last_k_ratio_i == "q_only":
                                imp_indices_doc = [cur_pos_final+k for k in range(doc_len)]
                            elif last_k_ratio == "q_only_plus":
                                
                                doc_imp_len = max(int((last_k_ratio_i)*input_len), 0)
                                for k in range(doc_imp_len-1,-1,-1):
                                    if k not in imp_indices:
                                        imp_indices_doc.append(cur_pos_final-k)
                                imp_indices_doc += [cur_pos_final+k for k in range(doc_len)]
                                
                            elif last_k_ratio_i == "first_doc_k_ratio":
                                doc_imp_len = max(int((first_doc_k_ratio)*doc_len), 0)
                                imp_indices_doc = [cur_pos_final+k for k in range(doc_imp_len)]
                            elif last_k_ratio_i == "last_one":
                                imp_indices_doc = [cur_pos_final+ doc_len -1]
                            else:
                                doc_imp_len = max(int((last_k_ratio_i)*input_len), 1)
                                for k in range(1, doc_imp_len):
                                    cur_idx = (input_len-k)
                                    if cur_idx not in imp_indices:
                                        imp_indices_doc.append(cur_idx)
                                imp_indices_doc.reverse()
                            #pdb.set_trace()
                        elif i > 0:
                            if first_doc_k_ratio == 0.0:
                                imp_indices_doc = []
                            else:
                                doc_imp_len = max(int(first_doc_k_ratio_list[i-1]*doc_len), 0)
                                imp_indices_doc = [cur_pos_final+k for k in range(doc_imp_len)]
                        
                        shifts.append(cur_pos)
                        imp_indices += imp_indices_doc
                        cur_pos += doc_len
                        
                        cur_pos_final = cur_pos + 1 # Be careful here

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
                       
                    
                    #Identify and replace HKE tokens
                    
                    #hke_ratio = 0.1
                    hke_tokens = []
                    for i in range(1, num_layer): #start with the second layer
                        #check and replace
                        v_dev = torch.sum((org_past_key_values[i][1][:,:,:-test_last_len].float()-chunk_past_key_values[i][1][:,:,:-test_last_len].float())**2, dim=[0,1,3])
                        hke_indices = torch.topk(v_dev, k=int(input_len*hke_ratio)).indices.cpu().tolist()
                        #chunk_past_key_values[i][0][:,:,hke_indices,:] = org_past_key_values[i][0][:,:,hke_indices,:].clone()
                        #chunk_past_key_values[i][1][:,:,hke_indices,:] = org_past_key_values[i][1][:,:,hke_indices,:].clone()
                        hke_tokens.append(hke_indices)
                    
                    #Identify and replace HFA tokens
                    
                    #hfa_ratio = hke_ratio
                    #hfa_tokens = []
                    
                    #for i in range(1, num_layer):
                    #    hfa_scores = torch.sum(org_post_attn[i][:,:1,:-test_last_len],dim=[0,1])
                    #    hfa_indices = torch.topk(hfa_scores, k=int(input_len*hfa_ratio)).indices.cpu().tolist()
                        #chunk_past_key_values[i][0][:,:,hfa_indices,:] = org_past_key_values[i][0][:,:,hfa_indices,:].clone()
                        #chunk_past_key_values[i][1][:,:,hfa_indices,:] = org_past_key_values[i][1][:,:,hfa_indices,:].clone()
                    #    hfa_tokens.append(hfa_indices)
                    
                    cdfs = []
                    #use_layer = 1
                    for i in range(num_layer-1):
                        cdfs.append(len(imp_indices))
                        for x in hke_tokens[i]:
                            if x not in imp_indices and i in use_layer:
                                imp_indices = [x] + imp_indices
                        #for x in hfa_tokens[i]:
                        #    if x not in imp_indices:
                        #        imp_indices = [x] + imp_indices
                    print(f"imp_indices:{imp_indices[:-len(s_end+q_ids)]}")
                    for i in range(len(cdfs)):
                        cdfs[i] = cdfs[i]/len(imp_indices)   
                        
                    real_recomp_ratios.append((len(imp_indices)-len(q_ids+s_end))/(input_len-len(q_ids+s_end)))
                    
                    
                    # compute overlapping ratio of HKE tokens and HFA tokens
                    '''
                    overlap_ratios = [] # each head,layer
                    for i in range(num_layer-1):
                        overlap = (len(hke_tokens[i]) + len(hfa_tokens[i]) - len([x for x in hke_tokens[i] if x in hfa_tokens[i]])) / input_len
                        overlap_ratios.append(overlap)
                    '''
                    #pdb.set_trace()
                    
                    # construct imp_indices
                    
                    chunk_past_key_values = [tuple(x) for x in chunk_past_key_values]
                    chunk_past_key_values = tuple(chunk_past_key_values)
                    del org_past_key_values
                    del past_key_values
                        
                    #check_post_attn = []
                    input_tensor = torch.tensor(input_ids, dtype=torch.long).view(1,-1).to(model.device)
                    with torch.no_grad():
                        res_toks = []
                        for i in range(max_gen_len):
                            if i == 0:
                                output_dict = model(input_tensor, 
                                                past_key_values=chunk_past_key_values,
                                                check=True,
                                                imp_indices=imp_indices, 
                                                check_layer_idx=check_layer)
                                #for j in range(num_layer):
                                #    check_post_attn.append(model.model.layers[j].self_attn.attention_weights_post[0,:,-test_last_len:].clone())
                                #    del model.model.layers[j].self_attn.attention_weights_post
                            else:
                                output_dict = model(input_tensor, past_key_values=past_key_values)
                            tok = torch.argmax(output_dict['logits'][:,-1,:])
                            past_key_values = output_dict['past_key_values']
                            if int(tok) == tokenizer.eos_token_id:
                                break
                            res_toks.append(int(tok))
                            input_tensor = tok.view(1,-1)
                        res_check = tokenizer.decode(res_toks)
                    
                    #attn_dev_list = []
                    #attn_dev_full_list = []
                    #for j in range(num_layer):
                    #    attn_dev = torch.mean(torch.sum((check_post_attn[j][:,:1,:].float()-org_post_attn[j][:,:1,:].float())**2, dim=2)**0.5).cpu().numpy()
                    #    attn_full_dev = torch.mean(torch.sum((check_post_attn[j][:,:,:].float()-org_post_attn[j][:,:,:].float())**2, dim=2)**0.5).cpu().numpy()
                    #    attn_dev_list.append(attn_dev)
                    #    attn_dev_full_list.append(attn_full_dev)
                    #print(f"Attention deviation: {np.mean(attn_dev_list)}")
                    #print(f"Attention deviation (full): {np.mean(attn_dev_full_list)}")
                    #attn_devs.append(np.mean(attn_dev_list))
                    #attn_full_devs.append(np.mean(attn_dev_full_list))
                        
                    #pdb.set_trace()
                    
                    del past_key_values
                    del chunk_past_key_values
                    
                    res_org = res_org.split("\n")[0] #FIXME: parse according to "."?
                    res_check = res_check.split("\n")[0]
                    
                    idx_i += 1
                    if True:#text_has_answer(answers, ex['ctxs'][0]['text']):
                        is_correct_org = max([compute_f1(res_org, answer, tokenizer) for answer in answers])
                        is_correct_check = max([compute_f1(res_check, answer, tokenizer) for answer in answers])
                        #is_correct_chunk = max([compute_f1(res_chunk, answer, tokenizer) for answer in answers])
                        is_correct_overlap = max([compute_f1(res_check, res_org, tokenizer)])
                        idx += 1
                        num_correct_org += is_correct_org
                        #num_correct_chunk += is_correct_chunk
                        num_correct_check += is_correct_check
                        num_correct_overlap += is_correct_overlap
                        
                        '''
                        output_check_diff = torch.sum((org_output-check_output)**2)**0.5
                        output_chunk_diff = torch.sum((org_output-chunk_output)**2)**0.5
                        
                        print("output_diff_check:", output_check_diff.cpu().numpy().sum())
                        print("output_diff_chunk:", output_chunk_diff.cpu().numpy().sum())
                        
                        print("dev_k_check:", np.mean(check_dev_k))
                        print("dev_k_chunk", np.mean(chunk_dev_k))
                        
                        print("dev_v_check:", np.mean(check_dev_v))
                        print("dev_v_chunk", np.mean(chunk_dev_v))
                        
                        print("dev_scores_pre_chunk:", np.mean(chunk_dev_scores_pre))
                        print("dev_scores_pre_check:", np.mean(check_dev_scores_pre))
                        
                        print("dev_scores_post_chunk:", np.mean(chunk_dev_scores_post))
                        print("dev_scores_post_check:", np.mean(check_dev_scores_post))
                        
                        #print("v_layer2_diff", chunk_dev_scores_post[2], "; Worsen:", is_correct_check<is_correct_org)
                        print("v_layer2_diff", (torch.sum((org_v[2][:,-100:].float()-chunk_v[2][:,-100:].float())**2)**0.5).cpu().numpy().sum(), "; Worsen:", is_correct_check<is_correct_org)
                        '''
                        
                        print("res_org:", res_org, is_correct_org)
                        print("res_check:", res_check, is_correct_check)
                        #print("res_chunk:", res_chunk)
                        print("answer in first:", text_has_answer(answers, ex['ctxs'][0]['text']))      
                        
                        
                        print(answers)
                        print(idx)
                        

                        #tq.set_description(f"Org: {num_correct_org / idx}; Check: {num_correct_check / idx}; Chunk: {num_correct_chunk / idx}; first doc k ratio:{first_doc_k_ratio}; lask k ratio: {last_k_ratio_i}; max_gen: {max_gen_len}")
                        tq.set_description(f"Org: {num_correct_org / idx}; Check: {num_correct_check / idx}; OL: {num_correct_overlap / idx}; first doc k ratio:{first_doc_k_ratio}; lask k ratio: {last_k_ratio_i}; max_gen: {max_gen_len}")
                    
                    #if get_model_parallel_rank()==0:
                    #    pdb.set_trace()
                    #torch.distributed.barrier()
                    
                    #pdb.set_trace()
                    
                    #if idx==30:
                        #pdb.set_trace()
                    #    break
                
                
                em_org = num_correct_org / idx
                em_check = num_correct_check / idx
                em_overlap = num_correct_overlap / idx
                avg_len = total_len/idx
                
                
                if output_dir is not None:
                    d = {"em_org": em_org,
                        "em_check": em_check,
                        "em_overlap": em_overlap,
                        "avg_len": avg_len,
                        "num_doc": num_docs,
                        "check_layer": check_layer,
                        "use_layer": use_layer,
                        "hke_ratio":hke_ratio,
                        "model": "70b",
                        "recomp_ratio":float(np.mean(real_recomp_ratios))}

                    with open("logs/compound1.json", "a") as f:
                        f.write(json.dumps(d) + "\n")


def load_dataset(dataset_path):
    print("Loading dataset:", dataset_path)
    with open(dataset_path) as f:
        return json.load(f)


def main():
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    #device = "cpu"
    with torch.no_grad():
        model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-70b-chat-hf",
                                            load_in_4bit=True,
                                            device_map='auto',
                                            torch_dtype=torch.float16,
                                            bnb_4bit_compute_dtype=torch.float16,
                                            use_cache=True)#.to(device) #Don't use to for load_in_8/4bit since the model has already been set to the correct devices and casted to the correct `dtype`.
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf")

    #pdb.set_trace()
    
    eval_dataset = load_dataset("datasets/wikimqa_e.json")
    
    for num_doc in [10]:
        evaluate_dataset(
            model, tokenizer,
            eval_dataset,
            num_docs=num_doc,
            output_dir="outputs/",
        )


if __name__ == '__main__':
    main()