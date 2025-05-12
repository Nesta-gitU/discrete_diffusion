import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer

import argparse
import os
import numpy as np
import torch as th
import torch.distributed as dist
from transformers import set_seed
import math
from src.metrics.utils import get_preprocessed_data, file_to_list

def main(args):
    set_seed(108)
    print("Mode:", args.mode)
    
    
    # Load model once.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path
    ).cuda()
    model.eval()
    print("Successfully loaded GPT-2 model for evaluation.")
    
    # Use a single file path (input_text is assumed to be a string)
    input_file = args.input_text
    
    # Load tokenizer and build a reverse mapping.
    #tokenizer = load_tokenizer(args.modality, args.experiment, os.path.split(args.model_path)[0])
    #instead load the tokenizer from gpt2-large? this is bad because 
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
    """
    text_samples = []
    if input_file.endswith('sde.json') or input_file.endswith('ode.json'):
        with open(input_file, 'r') as f:
            for line in f:
                print(line)
                print(line[0])
                text_samples.append(json.loads(line).split(' '))
    elif input_file.endswith('json'):
        with open(input_file, 'r') as f:
            for line in f:
                text_samples.append(json.loads(line).split(' '))
    else:
        with open(input_file, 'r') as f:
            for line in f:
                text_samples.append(line.strip().split())
    """
    text_samples = args.text_samples

    #do the below process for n_splits of the n_samples
    n_samples = len(text_samples)
    print(n_samples, args.std_split)
    print(f'Loaded {n_samples} samples from {input_file}')
    n_samples_per_split = n_samples / args.std_split
    text_samples_og = text_samples.copy()
    mean_loss_list = []
    for i in range(args.std_split):
        token_count = 0

        start_idx = int(i * n_samples_per_split)
        end_idx = int((i + 1) * n_samples_per_split)
        print(start_idx, end_idx)
        #print(n_samples_per_split)
        text_samples = text_samples_og[start_idx:end_idx]
        #print(text_samples)

        agg_loss = []
        for x in text_samples:
            try:
                tokenized_x = tokenizer.encode(x)
            except KeyError as e:
                print(f"Warning: token not found in tokenizer: {e}. Skipping sample.")
                continue
            tokenized_x = torch.LongTensor(tokenized_x).cuda()
            labels = tokenized_x.clone()
            #labels[labels == tokenizer.encode('PAD')[0]] = -100
            labels[0] = -100 #ignore the start token 
            model_output = model(tokenized_x, labels=labels)
            loss = model_output.loss.item()
            #agg_loss.append(loss * tokenized_x.numel())  # accumulate total NLL
            #token_count += tokenized_x.numel()
            ppl_story   = math.exp(loss)
            agg_loss.append(ppl_story)                    # store per-text PPL
    
        #total_nll   = sum(agg_loss)
        #total_tok   = token_count
        #mean_loss   = total_nll / total_tok                   # corpus-wide mean NLL
        #perp        = math.exp(mean_loss)
        #mean_loss_list.append(perp)
        ppl_split = sum(agg_loss) / len(agg_loss)        # arithmetic mean
        mean_loss_list.append(ppl_split)
    
    mean_loss = np.mean(mean_loss_list)
    std_loss = np.std(mean_loss_list)

    
    print(f'\nThe mean loss is {mean_loss} for {input_file}')
    print(f'The standard deviation is {std_loss} for {input_file}')
    print('-' * 50)
    
    return mean_loss, std_loss

if __name__ == '__main__':
    import torch
    with torch.no_grad():
        main()
