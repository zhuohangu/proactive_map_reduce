from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import argparse
import pandas as pd
import numpy as np

chencherry = SmoothingFunction()


default_model_name = "mistralai/Mistral-7B-Instruct-v0.3"

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--model_name", type=str, default=default_model_name)
    args.add_argument("--num_steps", type=int, default=5)
    args = args.parse_args()
    return args

args = get_args()

file_baseline = f'opus_short_outputs_en_fr/baseline_{args.model_name.split("/")[1]}.json'
file_proactive = f'opus_short_outputs_en_fr/proactive2_{args.num_steps}_steps_{args.model_name.split("/")[1]}.json'

df_baseline = pd.read_json(file_baseline, lines=True)
df_proactive = pd.read_json(file_proactive, lines=True)


baseline_bleu = df_baseline["bleu"].tolist()
baseline_f1 = df_baseline["f1"].tolist()
baseline_e2e_delays = df_baseline["e2e_delay"].tolist()
baseline_decoding_speed = df_baseline["decoding_speeds"].tolist()

baseline_doc_chunk_ids_length = df_baseline["doc_chunk_ids_length"].tolist()
baseline_doc_chunk_ids_length = [item for sublist in baseline_doc_chunk_ids_length for item in sublist]

baseline_inputs_prefill_times = df_baseline["inputs_prefill_times"].tolist()
baseline_inputs_prefill_times = [item for sublist in baseline_inputs_prefill_times for item in sublist]

baseline_decode_times = df_baseline["decode_times"].tolist()
baseline_decode_times = [item for sublist in baseline_decode_times for item in sublist]

proactive_bleu = df_proactive["bleu"].tolist()
proactive_f1 = df_proactive["f1"].tolist()
proactive_e2e_delays = df_proactive["e2e_delay"].tolist()

proactive_doc_chunk_ids_length = df_proactive["doc_chunk_ids_length"].tolist()

proactive_inputs_prefill_times = df_proactive["inputs_prefill_times"].tolist()

proactive_decode_times = df_proactive["decode_times"].tolist()


speedup = []



print(f"Average BASELINE BLEU SCORE: {np.mean(baseline_bleu)}")
print(f"Average BASELINE f1 SCORE: {np.mean(baseline_f1)}")
print(f"Average PROACTIVE BLEU SCORE: {np.mean(proactive_bleu)}")
print(f"Average PROACTIVE f1 SCORE: {np.mean(proactive_f1)}")

# Calculate baseline pipeline time
speed_ups = []
tokens_per_sec = np.mean(baseline_decoding_speed)
print(f"Tokens per second: {tokens_per_sec}")
for i in range(len(proactive_bleu)):
    t_L = baseline_doc_chunk_ids_length[i] / tokens_per_sec
    baseline_e2e_delay = t_L + baseline_inputs_prefill_times[i] + baseline_decode_times[i]
    
    # print(f"BASELINE E2E DELAY: {baseline_e2e_delay}")

    # Calculate proactive pipeline time
    proactive_e2e_delay = proactive_doc_chunk_ids_length[i][0] / tokens_per_sec
    for j in range(0, args.num_steps):
        if j == args.num_steps-1:
            proactive_e2e_delay += proactive_inputs_prefill_times[i][j] + proactive_decode_times[i][j]
        elif (proactive_e2e_delay + (proactive_doc_chunk_ids_length[i][j+1] / tokens_per_sec)) > (proactive_e2e_delay + proactive_inputs_prefill_times[i][j] + proactive_decode_times[i][j]):
            proactive_e2e_delay += (proactive_doc_chunk_ids_length[i][j+1] / tokens_per_sec)
        else:
            proactive_e2e_delay += (proactive_inputs_prefill_times[i][j] + proactive_decode_times[i][j])
    # print(f"PROACTIVE E2E DELAY: {proactive_e2e_delay}")

    speed_ups.append(baseline_e2e_delay / proactive_e2e_delay)

print(f"SPEEDUPs: {speed_ups}")
print(f"Average SPEEDUP: {np.mean(speed_ups)}")

