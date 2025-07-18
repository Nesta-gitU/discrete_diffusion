import torch
import wandb 
import matplotlib.pyplot as plt
import numpy as np

from .differential_equations import sample_loop
from .methods import top_k, top_p, argmax_sample
import json

from src.metrics.entropy import get_per_token_entropy

from src.metrics.entropy import get_per_token_entropy
from src.models.nfdm.components.forward_process import Sqrt
from src.models.ndm.components.gamma import GammaVDM
from src.sampling.animation import make_video

import os


def ani(model):
    with torch.no_grad():

        embeddings = model.pred.model.word_embedding.weight
        print(embeddings.shape, "embedding_shape") #[vocab_size, emb_dim]

      
        new_embds = torch.nn.functional.normalize(embeddings, dim=1)  #  [vocab_size, emb_dim]


        cosine_similarities = new_embds @ new_embds.T  # [vocab_size, vocab_size]

        vocab_size = new_embds.size(0)
        mask = ~torch.eye(vocab_size, dtype=torch.bool, device=embeddings.device)  # Mask for non-diagonal elements
        pairwise_cosines = cosine_similarities[mask]  # Extract only off-diagonal elements
        
        # Compute ANI
        ani = pairwise_cosines.mean().item()
        print("ani is:")
        print(ani, "------------------------------------")             



def sample_code(model, 
                tokenizer, 
                batch_size,
                block_size, 
                hidden_size, 
                out_dir,
                model_base_name,
                sampling_mode,
                n_steps, 
                debug, 
                clamping, 
                do_top_k, 
                k, 
                do_top_p, 
                p, 
                temperature, 
                compute_ani,
                num_samples,
                animate,
                time_sampler_args): #add clamping as an option, also add sampling instead of argmax as an option later, 

        if True:
            ani(model)

        if hasattr(model, 'gamma'):
            plot_gamma(model,out_dir,model_base_name,batch_size,block_size,hidden_size)
        
        batches_needed = num_samples // batch_size
        w = []
        total_entropy = 0
        for i in range(batches_needed):
            latent, words, path = sample_from_diffusion(model = model, batch_size=batch_size, block_size=block_size, \
                hidden_size = hidden_size, _n_steps = n_steps, clamping=clamping, do_top_k=do_top_k, k=k,
                do_top_p= do_top_p, p =p, temperature = temperature, sampling_mode=sampling_mode, tokenizer=tokenizer, time_sampler_args=time_sampler_args)

            #print(latent_sde.shape, "sde this should have the shape [batch_size, block_size, hidden_size]")
            #print(latent_ode.shape, "ode this should have the shape [batch_size, block_size, hidden_size]")
            
            if True:
               
                print("visualizing embedding matrix")
                visualize_embedding_matrix(model)

                #visualize the path somehow
                print("visualizing path")
                visualize_path(model, path, tokenizer, mode=sampling_mode)
            
            if animate:
                #visualize the path with an animation 
                print("visualizing path with animation")
                # okay so the trace holds a list of matrices for each of the
                trace_words = []

                for i in range(path.shape[0]):
                    decoded = model.pred.model.get_logits(path[i][0])
                    decoded = decoded.argmax(dim=-1).squeeze(-1)
                    tokens = tokenizer.decode(decoded)
                    trace_words.append(tokens.split())
                print(trace_words, "trace words")
                make_video(trace_words, fps=6, path=os.path.join(out_dir, f"{model_base_name}.samples_{sampling_mode}.mp4"), tween=1)



            if latent is not None:
                total_entropy=0
                #logits = model.pred.model.get_logits(latent)
                #total_entropy += get_entropy(logits, words, tokenizer, sampling_mode)
                
            else:
                logits = None
           
            
            if words is None:
                w = None
            else:
                w.extend(idx_to_words(words, tokenizer))
            
            out_path = os.path.join(out_dir, f"{model_base_name}.samples_{sampling_mode}.json")
        
        print(f"Per-token entropy of the generated samples for", sampling_mode ,":", total_entropy/batches_needed ,"---------------------------------------------------------")
        
        if w is not None:
            with open(out_path, 'w') as f:
                for text in w:
                    # Wrap each text sample in a list and write it as a JSON string on a new line
                    f.write(json.dumps(text) + "\n")

        return total_entropy/batches_needed

def sample_from_diffusion(model, batch_size, block_size, hidden_size, _n_steps=100, clamping=False, do_top_k=False, 
                        k=10, do_top_p=False, p=0.9, temperature=1.0, sampling_mode='marginals', tokenizer=None, time_sampler_args=None):
    assert top_k != top_p, "top_k and top_p cannot be used together"
    # sample batch size random z's, these must have the shape equal to our datapoints so that is [batch_size, block_size, n_embed]

    #z = torch.randn(torch.Size(batch_size, block_size, hidden_size))
    z = torch.randn(batch_size, block_size, hidden_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = z.to(device)
    model.to(device)
   
    solved, path = sample_loop(z, 1, 0, _n_steps, model, sampling_mode, time_sampler_args,  clamping)
    #decode the sde solve back into words 
    logits = model.pred.model.get_logits(solved)
    
    
    if do_top_k:
        indices = top_k(logits, k, temperature, tokenizer, remove_unk=False)
    elif do_top_p:
        indices = top_p(logits, p, temperature, tokenizer, remove_unk=False)
    else:
        indices = argmax_sample(logits, tokenizer, remove_unk=False)

    return solved, indices, path

def visualize_embedding_matrix(model):
    embeddings = model.pred.model.word_embedding.weight
    print(embeddings.shape, "embedding shape")

    # Assuming `embeddings` is your matrix
    # If using PyTorch, convert it to a NumPy array first
    embeddings_np = embeddings.detach().cpu().numpy()

    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.imshow(embeddings_np, cmap='viridis_r', aspect='auto')  # 'viridis' is a color map with darker colors for smaller values
    plt.colorbar(label='Value')
    plt.title('Embedding Matrix Visualization')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.savefig("embedding_matrix.png")
    plt.close()


def visualize_latent(pl_module, words, tokenizer):
    
    if words is None:
        pass
    else:
        first_latent = words[0] #shape [block_size, hidden_size]
        print(first_latent.shape, "first latent shape")

        for i in range(first_latent.shape[0]):
            latent_vector = first_latent[i]
            print("the latent,", i, " ", latent_vector)
            decoded = pl_module.model.decoder(latent_vector, pl_module.model.encoder.embedding.weight)
            decoded = decoded.argmax(dim=-1).squeeze(-1)
            print(decoded.shape, "decoded shape")
            print(tokenizer.decode([decoded.item()]))


def visualize_path(model, path, tokenizer, mode):
    # for the sde path print all the intermediate steps in a row for the first token
    """
    path = [z]
    for t in tqdm(tt):
        t = t.expand(bs, 1,1)
        
        if mode == 'sde':
            f, g = sde_drift(z, t, module)
        else:
            f, g = ode_drift(z, t, module)

        w = torch.randn_like(z)
        z = z + f * dt + g * w * dt_2
        
        path.append(z)
        
    return z, torch.stack(path)
    """
    if path is None:
        pass
    else:
        for i in range(path.shape[0]):
            print("type", path[i][0])
            #also print the corersponding word
            #decoded = #pl_module.model.decoder(sde_path[i][0], pl_module.model.encoder.embedding.weight)
            decoded = model.pred.model.get_logits(path[i][0])
            #print(decoded.shape, "decoded")
            decoded = decoded.argmax(dim=-1).squeeze(-1)
            try:
                tokens = tokenizer.decode(decoded)
            except:
                tokens = "error"

            print(mode, " ", tokens)
    
def get_entropy(logits, words, tokenizer, name=None):
    padding_token = 'PAD'
    padding_index = tokenizer.vocab_dict.get(padding_token, None)

    if padding_index is None:
        print("Padding token 'PAD' not found in the tokenizer vocabulary. Setting padding mask to None.")
        padding_mask = None
    else:
        # Create the padding mask (1 for non-padding tokens, 0 for padding tokens)
        padding_mask = (words != padding_index).float()
    
    # print the paddingmask for the first sample
    print(padding_mask[0], "padding mask")
    
    #compute the entropy of the generated samples
    entropy = get_per_token_entropy(logits, padding_mask)
    return entropy

def idx_to_words(index, tokenizer) -> list:
    decoded_texts = []
    for sequence in index:
        try:
            tokens = tokenizer.decode(sequence)
        except Exception as e:
            print(f"Error decoding tokens: {e}")
            tokens = "error"
        decoded_texts.append(tokens)
    print('--------final text--------------------')
    print(decoded_texts[0])
    return decoded_texts



def plot_gamma(model, out_dir, model_base_name, batch_size, block_size, hidden_size):

    # Create the directory if it doesn't exist
    outpath = os.path.join(out_dir, f"{model_base_name}")
    os.makedirs(outpath, exist_ok=True)
    
    #also init a new gamma from scratch and plot that too see what it was originally
    gamma_og = GammaVDM()

    #z = torch.randn(300, block_size, hidden_size) #this means each t has a different z, but I want to see all t for some z 
    z = torch.randn(1, block_size, hidden_size)
    #z = z.expand(300, -1, -1) 
    #although the fact that for a different t at each time we get a bunch of smooth lines seems like a really bad sign to me, means z has almost no effect?

    with torch.no_grad():
        model.to("cpu")
        t = torch.linspace(0, 1, 300)[:, None].to(model.pred.model.word_embedding.weight.device)
        t=t.unsqueeze(-1)

        
        context = model.context.sample_context(z) #slightly incorrect if using NN but with VAE its fine
        if context is None:
            pass
        else:
            context = context.expand(300, -1, -1) 
        
        if context is None:
            gmm, _ = model.gamma(t)
        else:
            gmm, _ = model.gamma(t, context)
        
        alpha_2 = model.gamma.alpha_2(gmm)
        sigma_2 = model.gamma.sigma_2(gmm)
        alpha = alpha_2 ** 0.5
        sigma = sigma_2 ** 0.5

        gmm = gmm.squeeze(-1)
        alpha = alpha.squeeze(-1)
        sigma = sigma.squeeze(-1)

        #also get alpha and sigma from the sqrt function and plot them in the same graph as the other alpha and sigma, so make three plots
        sqrt = Sqrt()
        _, sigma_sqrt, alpha_sqrt = sqrt(torch.tensor(1.), t)
        alpha_sqrt = alpha_sqrt.squeeze(-1)
        sigma_sqrt = sigma_sqrt.squeeze(-1)

        #also get alpha and sigma and gmm for the original gamma
        gmm_og, _ = gamma_og(t)
        alpha_2_og = gamma_og.alpha_2(gmm_og)
        sigma_2_og = gamma_og.sigma_2(gmm_og)
        alpha_og = alpha_2_og ** 0.5
        sigma_og = sigma_2_og ** 0.5

        gmm_og = gmm_og.squeeze(-1)
        alpha_og = alpha_og.squeeze(-1)
        sigma_og = sigma_og.squeeze(-1)

        #print the order of the sequence wise gamma. 
        print("the order gamma is: ")
        #shape of gmm in the case where an order is relevant is [300, 64, 1]
        #now it is not data dependent so we can ignore batch dim and just look at the order at 0.5 I guess
        gmm_for_order = gmm[150]
        sorted_gmm = torch.argsort(gmm_for_order, descending=True) #high to low so we know what is being maksed out first 
        print(sorted_gmm, "sorted gmm")
        # [64, 1] I want the order where I print (ggm_dim, )


        try:
            plt.plot(alpha)
            plt.plot(alpha_sqrt)
            plt.plot(alpha_og, label="alpha_og")
            plt.legend(["alpha", "alpha_sqrt", "alpha_og"])
            plt.savefig(f"{outpath}/alpha_plot.png")
            plt.close()

            plt.plot(sigma)
            plt.plot(sigma_sqrt)
            plt.plot(sigma_og, label="sigma_og")
            plt.legend(["sigma", "sigma_sqrt", "sigma_og"])
            plt.savefig(f"{outpath}/sigma_plot.png")
            plt.close()

            plt.plot(gmm)
            plt.plot(gmm_og, label="gamma_og")
            plt.savefig(f"{outpath}/gamma_plot.png")
            plt.close()
        except Exception as e:
            pass
