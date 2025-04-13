import json
import numpy as np


def get_preprocessed_data(split, datamodule, tokenizer, num_samples):
    if split == 'train':
        data = datamodule.train_dataloader()
    elif split == 'val':
        data = datamodule.val_dataloader()    
    elif split == 'test':
        data = datamodule.test_dataloader()

    decoded_texts = []
    for batch in data:
        ids = batch
        #print("ids", ids)
        
        for sequence in ids:
            tokens = tokenizer.decode(sequence)
            #print("decoded tokens", tokens)
            decoded_texts.append(tokens)
        
        if len(decoded_texts) > num_samples:
            break

    decoded_texts = decoded_texts[:num_samples]
    return decoded_texts    

def file_to_list(text_path, datamodule, tokenizer, setting):
    text_samples = []
    with open(text_path, 'r') as f:
        if text_path.endswith('.json'):
            for line in f:
                text_samples.append(line[1:-2])
        else:
            for line in f:
                text_samples.append(line)
    
    # Remove trailing pad tokens from each generated sample.
    # Assumes pad tokens are represented as "<pad>"
    def remove_pad_tokens(sample, pad_token='PAD'):
        #print("input to pad token removal", sample)
        if isinstance(sample, list):
            # If sample is a list of tokens, remove trailing pad tokens
            while sample and sample[-1] == pad_token:
                sample.pop()
            #print("removed pad tokens", sample)
            return " ".join(sample)
        
        elif isinstance(sample, str):
            # If sample is a string, split it into tokens (by whitespace)
            tokens = sample.split()
            #print("split tokens",tokens)
            while tokens and tokens[-1] == pad_token:
                tokens.pop()
            return " ".join(tokens)
        
        else:
            #print("nothing was done somehow", sample)
            return sample

    # Apply pad removal to each generated sample
    
    n_generated_samples = len(text_samples)
    if setting == 'reference_mode':
        split = 'val'
    else:
        split = 'test'
    val_samples = get_preprocessed_data(split, datamodule, tokenizer, n_generated_samples)
    
    print("computing mauve for {} samples".format(n_generated_samples))
    print("againts {} samples".format(len(val_samples)))

    # Compute Mauve score
    all_texts_list = [remove_pad_tokens(sample) for sample in text_samples] 
    human_references = [remove_pad_tokens(sample) for sample in val_samples]

    # print one human reference and one generated sample to see if pad toekns where removed
    print("human reference", human_references[0])
    print("generated sample", all_texts_list[0])

    return all_texts_list, human_references

def metric_to_std(all_texts_list, human_references, metric_function, std_split, num_gen_samples):
    # just make the input to all metric computer functions the same, but only use the needed inputs
    #text is generated
    #val is human

    # text path is a json file    
    split_length = num_gen_samples / std_split
    
    metric_list = []
    for i in range(std_split):
        text_samples_split = all_texts_list[int(i*split_length):int((i+1)*split_length)]
        val_samples_split = human_references[int(i*split_length):int((i+1)*split_length)]

        metric = metric_function(all_texts_list=text_samples_split, human_references=val_samples_split)
        metric_list.append(metric)
    
    metric_array = np.array(metric_list)
    mean_metric = np.mean(metric_array)
    std_metric = np.std(metric_array)

    return mean_metric, std_metric