import mauve 

def print_mauve(all_texts_list, human_references):

    out = mauve.compute_mauve(p_text=all_texts_list, q_text=human_references) #keeping max seq length as default is honestly fine sqe len still a bit too short for reliable mauve but okay
    #print(out) -> potentially mauve star could be a better metric but my baselines use mauve so lets stick with that. 
    mauve_score = out.mauve
    mauve_frontier_integral = out.frontier_integral
    mauve_score_star = out.mauve_star
    mauve_frontier_integral_star = out.frontier_integral_star

    return {"1": mauve_score, "2":mauve_score_star, "2":mauve_frontier_integral, "4":mauve_frontier_integral_star}