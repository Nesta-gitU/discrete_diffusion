import mauve 

def print_mauve(all_texts_list, human_references):

    out = mauve.compute_mauve(p_text=all_texts_list, q_text=human_references, max_text_length=64)
    #print(out) -> potentially mauve star could be a better metric but my baselines use mauve so lets stick with that. 
    mauve_score = out.mauve

    return mauve_score