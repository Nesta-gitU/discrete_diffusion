import mauve 
import json
import numpy as np
from improved_diffusion.text_datasets import load_data_text
from improved_diffusion.rounding import load_models

class DotDict(dict):
    """Dictionary with dot notation access to attributes."""
    def __getattr__(self, key):
        try:
            value = self[key]
            # Optionally, recursively convert inner dictionaries.
            if isinstance(value, dict):
                value = DotDict(value)
            return value
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")


def get_preprocessed_data(split, training_args, num_samples):
    training_args_ = DotDict(training_args)
       training_args['modality'], 
        training_args['experiment'], 
        training_args['model_name_or_path'], 
        training_args['in_channel'],
        training_args["checkpoint_path"]
    )
    if training_args['modality'] == 'book' or training_args['use_bert_tokenizer'] == 'yes':
        rev_tokenizer = tokenizer  # BERT tokenizer BPE.
    else:
        rev_tokenizer = {v: k for k, v in tokenizer.items()}

    data = load_data_text(
        data_dir=training_args['data_dir'],
        batch_size=training_args['batch_size'],
        image_size=training_args['image_size'],
        class_cond=training_args['class_cond'],
        data_args=training_args_,
        task_mode=training_args['modality'],
        padding_mode=training_args['padding_mode'],  # block, pad
        split='test',
        load_vocab=rev_tokenizer,
        model=model2,
    )model2, tokenizer = load_models(
     

    decoded_texts = []
    for batch in data:
        ids = batch[-1]['input_ids']
        
        for sequence in ids:
            tokens = " ".join([tokenizer[x.item()] for x in sequence])
            decoded_texts.append(tokens)
        if len(decoded_texts) > num_samples:
            break

    decoded_texts = decoded_texts[:num_samples]
    return decoded_texts    


def print_mauve(text_path, modality, std_split, setting, training_args):
    # text path is a json file
    text_samples = []
    with open(text_path, 'r') as f:
        for line in f:
            print("line", line)
            text_samples.append(line)
    
    # Remove trailing pad tokens from each generated sample.
    # Assumes pad tokens are represented as "<pad>"
    def remove_pad_tokens(sample, pad_token='PAD'):
        if isinstance(sample, list):
            # If sample is a list of tokens, remove trailing pad tokens
            while sample and sample[-1] == pad_token:
                sample.pop()
            return sample
        elif isinstance(sample, str):
            # If sample is a string, split it into tokens (by whitespace)
            tokens = sample.split()
            print("split tokens",tokens)
            while tokens and tokens[-1] == pad_token:
                tokens.pop()
            return " ".join(tokens)
        
        else:
            return sample

    # Apply pad removal to each generated sample
    
    n_generated_samples = len(text_samples)
    if setting == 'reference_mode':
        split = 'val'
    else:
        split = 'test'
    val_samples = get_preprocessed_data(split=split, training_args=training_args, num_samples=n_generated_samples)
    
    print("computing mauve for {} samples".format(n_generated_samples))
    print("againts {} samples".format(len(val_samples)))

    # Compute Mauve score
    text_samples = [remove_pad_tokens(sample) for sample in text_samples] 
    val_samples = [remove_pad_tokens(sample) for sample in val_samples]
    
    split_length = n_generated_samples / std_split
    
    mauve_list = []
    for i in range(std_split):
        text_samples_split = text_samples[int(i*split_length):int((i+1)*split_length)]
        print(text_samples_split)
        val_samples_split = val_samples[int(i*split_length):int((i+1)*split_length)]
        print(val_samples_split)

        out = mauve.compute_mauve(p_text=text_samples_split, q_text=val_samples_split, max_text_length=64)
        mauve_score = out.mauve
        mauve_list.append(mauve_score)
    
    mauve_array = np.array(mauve_list)
    mean_mauve = np.mean(mauve_array)
    std_mauve = np.std(mauve_array)

    return mean_mauve, std_mauve


