import torch
from evaluate import load


def compute_perplexity(all_texts_list, human_references, model_id='gpt2-large'):
    print(all_texts_list)
    torch.cuda.empty_cache() 
    perplexity = load("perplexity", module_type="metric", keep_in_memory=True)
    results = perplexity.compute(predictions=all_texts_list, model_id=model_id, device='cuda', add_start_token=True)
    return results['mean_perplexity']