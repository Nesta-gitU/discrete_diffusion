import json
import numpy as np

from typing import List, Union
import re


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
    def postprocess_generation(
        texts: Union[str, List[str]],
        start_token: str = "<s>",
        end_token: str = "</s>"
    ) -> Union[str, List[str]]:
        """
        Cleans up model-generated text by:
        1. Removing specified start/end tokens
        2. Stripping leading/trailing whitespace
        3. Removing spaces before dots and commas
        4. Removing spaces before and after apostrophes

        Args:
        texts:      A single string or list of strings to clean.
        start_token: Token to remove at the beginning.
        end_token:   Token to remove at the end.

        Returns:
        Cleaned string or list of strings.
        """
        pad_token_count = 0
        def clean(s: str) -> str:
            # 1) Remove start/end tokens
            s = s.replace(start_token, "").replace(end_token, "")
            # 2) Remove spaces before dots and commas
            s = re.sub(r"\s+([.,!?;:])", r"\1", s)
            # 3) Remove spaces around apostrophes
            s = re.sub(r"\s*'\s*", r"'", s)
            # 4) Strip any remaining leading/trailing whitespace
            return s.strip()
        
        def remove_pad_tokens(sample, pad_token='PAD'):
            nonlocal pad_token_count
        #print("input to pad token removal", sample)
        
            if isinstance(sample, list):
                # If sample is a list of tokens, remove trailing pad tokens
                while sample and sample[-1] == pad_token:
                    sample.pop()
                    pad_token_count += 1
                #print("removed pad tokens", sample)
                return " ".join(sample)
            
            elif isinstance(sample, str):
                # If sample is a string, split it into tokens (by whitespace)
                tokens = sample.split()
                #print("split tokens",tokens)
                while tokens and tokens[-1] == pad_token:
                    tokens.pop()
                    pad_token_count += 1  
                return " ".join(tokens)
            
            else:
                #print("nothing was done somehow", sample)
                return sample

        avg_unks = 0
        if isinstance(texts, str):
            for word in texts.split():
                if word == "UNK":
                    avg_unks += 1
            print("average unks", avg_unks)
            return clean(remove_pad_tokens(texts)), avg_unks, pad_token_count
        else:
            # If texts is a list, process each string
            for text in texts:
                for word in text.split():
                    if word == "UNK":
                        avg_unks += 1
            
            print("average unks", avg_unks/len(texts))
            return [clean(remove_pad_tokens(t)) for t in texts], avg_unks/len(texts), pad_token_count/len(texts)


    # Apply pad removal to each generated sample
    
    n_generated_samples = len(text_samples)
    if setting == 'reference_mode':
        split = 'val'
    else:
        split = 'test'
    val_samples = get_preprocessed_data(split, datamodule, tokenizer, n_generated_samples)
    train_samples = get_preprocessed_data('train', datamodule, tokenizer, n_generated_samples)
    
    print("computing mauve for {} samples".format(n_generated_samples))
    print("againts {} samples".format(len(val_samples)))

    # Compute Mauve score
    all_texts_list, unk, pad = postprocess_generation(text_samples, start_token="START", end_token="END") #[remove_pad_tokens(sample) for sample in text_samples] 
    human_references, _, _  = postprocess_generation(val_samples, start_token="START", end_token="END") #[remove_pad_tokens(sample) for sample in val_samples]
    human_references_train, _, _ = postprocess_generation(train_samples, start_token="START", end_token="END") #[remove_pad_tokens(sample) for sample in train_samples]

    # print one human reference and one generated sample to see if pad toekns where removed
    print("human reference", human_references[0])
    print("generated sample", all_texts_list[0])

    return all_texts_list, human_references, human_references_train, unk, pad

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
    
    if isinstance(metric_list[0], dict):
        mean_metric = []
        std_metric = []
        for key in metric_list[0].keys():
            cur_list = [m[key] for m in metric_list]
            mean_value = np.mean(cur_list)
            std_value = np.std(cur_list)
            mean_metric.append(mean_value)
            std_metric.append(std_value)
    else:
        metric_array = np.array(metric_list)
        mean_metric = np.mean(metric_array)
        std_metric = np.std(metric_array)

    return mean_metric, std_metric