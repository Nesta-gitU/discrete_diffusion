import torch as th

def token_discrete_loss(x_t, get_logits, input_ids, sum=False):
    
    reshaped_x_t = x_t
    logits = get_logits(reshaped_x_t)  # bsz, seqlen, vocab
    # print(logits.shape)
    loss_fct = th.nn.CrossEntropyLoss(reduction='none')
    decoder_nll = loss_fct(logits.view(-1, logits.size(-1)), input_ids.view(-1)).view(input_ids.shape)
    # print(decoder_nll.shape)

    return decoder_nll