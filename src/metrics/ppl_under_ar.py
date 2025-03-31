import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer

import argparse
import os
import numpy as np
import torch as th
import torch.distributed as dist
from transformers import set_seed


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
    tokenizer.add_tokens("PAD")	
    model.resize_token_embeddings(len(tokenizer))
    
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

    #do the below process for n_splits of the n_samples
    n_samples = len(text_samples)
    print(f'Loaded {n_samples} samples from {input_file}')
    n_samples_per_split = n_samples / args.std_split

    mean_loss_list = []
    for i in range(args.std_split):

        start_idx = int(i * n_samples_per_split)
        end_idx = int((i + 1) * n_samples_per_split)
        text_samples = text_samples[start_idx:end_idx]

        agg_loss = []
        for x in text_samples:
            try:
                tokenized_x = tokenizer.encode(x)
            except KeyError as e:
                print(f"Warning: token not found in tokenizer: {e}. Skipping sample.")
                continue
            tokenized_x = torch.LongTensor(tokenized_x).cuda()
            labels = tokenized_x.clone()
            labels[labels == tokenizer.encode('PAD')[0]] = -100
            model_output = model(tokenized_x, labels=labels)
            loss = model_output.loss.item()
            agg_loss.append(loss)
    
        mean_loss = torch.tensor(agg_loss).mean().item() if agg_loss else float('nan')
        mean_loss_list.append(mean_loss)
        
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
