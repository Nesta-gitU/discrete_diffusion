import torch
import wandb 
import matplotlib.pyplot as plt
import numpy as np

from src.utils.differential_equations import sde_drift, solve_de
from src.sampling.methods import top_k, top_p, argmax_sample
import json

import os

def ani(pl_module, datamodule, logger, tokenizer):
    #get embeddings from the encoder
    embeddings = pl_module.model.encoder.embedding.weight

    # Normalize the embeddings to unit vectors
    new_embds = torch.nn.functional.normalize(embeddings, dim=1)  # Shape remains [vocab_size, emb_dim]
    #print(embeddings.shape, "embedding_shape") [vocab_size, emb_dim]

    # Compute cosine similarities (dot products of normalized embeddings)
    cosine_similarities = new_embds @ new_embds.T  # Shape: [vocab_size, vocab_size]

    # Exclude self-similarities by subtracting the diagonal
    vocab_size = new_embds.size(0)
    mask = ~torch.eye(vocab_size, dtype=torch.bool, device=embeddings.device)  # Mask for non-diagonal elements
    pairwise_cosines = cosine_similarities[mask]  # Extract only off-diagonal elements
    # Compute ANI
    ani = pairwise_cosines.mean().item()
    print("ani is:")
    print(ani, "------------------------------------")             



def sample_code(pl_module, datamodule, logger, get_sde, get_ode, n_steps, batch_size, debug, clamping, do_top_k, k, do_top_p, p, temperature, compute_ani): #add clamping as an option, also add sampling instead of argmax as an option later, 
        tokenizer = datamodule.tokenizer

        if compute_ani:
            ani(pl_module, datamodule, logger, tokenizer)

        latent_sde, latent_ode, words_sde, words_ode, sde_path, ode_path = sample_from_diffusion(pl_module, batch_size, get_sde, get_ode, n_steps, clamping, do_top_k, k, do_top_p, p, temperature)

        #print(latent_sde.shape, "sde this should have the shape [batch_size, block_size, hidden_size]")
        #print(latent_ode.shape, "ode this should have the shape [batch_size, block_size, hidden_size]")

        if debug:
            #visualize the embedding matrix with color coding 
            print("visualizing embedding matrix")
            visualize_embedding_matrix(pl_module)

            #visualize the path somehow
            print("visualizing path")
            visualize_path(pl_module, sde_path, ode_path, logger, tokenizer)

            #take the first latent matrix in the batch, print each time the word vector and the word to which it was decoded
            # #maybe also the original word vector
            #print("visualizing latent")
            #visualize_latent(pl_module, latent_sde, latent_ode, tokenizer) 

        tokenizer = datamodule.tokenizer

        if words_sde is None:
            w_sde = None
        else:
            w_sde = idx_to_words(words_sde, tokenizer)
        if words_ode is None:
            w_ode = None
        else:
            w_ode = idx_to_words(words_ode, tokenizer)
        
        output_file = "./discrete_diffusion/output/samples.json"

        samples = {
            "text_ode": str(w_ode),
            "text_sde": str(w_sde)
        }
        #create the dir
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(samples, f)

def sample_from_diffusion(module, batch_size, get_sde=True, get_ode=True, _n_steps=100, clamping=False, do_top_k=False, k=10, do_top_p=False, p=0.9, temperature=1.0):
    assert top_k != top_p, "top_k and top_p cannot be used together"
    # sample batch size random z's, these must have the shape equal to our datapoints so that is [batch_size, block_size, n_embed]
    # I should be able to get the from the input_size and block_size of the transformer model
    block_size = module.model.pred.model.block_size
    if module.model.pred.model.small_input_size is not None:
        hidden_size = module.model.pred.model.small_input_size
    else:
        hidden_size = module.model.pred.model.hidden_size

    #z = torch.randn(torch.Size(batch_size, block_size, hidden_size))
    z = torch.randn(batch_size, block_size, hidden_size)
    z = z.to(module.device)

    sde_indices = None
    ode_indices = None

    if get_sde:
        sde_solved, sde_path = solve_de(z, 1, 0, _n_steps, module, 'sde', clamping)
        #decode the sde solve back into words 
        sde_logits = module.model.decoder(sde_solved, module.model.encoder.embedding.weight)
    else:
        sde_solved = None
        sde_path = None
        sde_logits = None
        
        
    if get_ode:
        ode_solved, ode_path = solve_de(z, 1, 0, _n_steps, module, 'ode', clamping)
        #decode the ode solve back into words
        ode_logits = module.model.decoder(ode_solved, module.model.encoder.embedding.weight)
    else:
        ode_solved = None
        ode_path = None
        ode_logits = None
    
    if do_top_k:
        sde_indices = top_k(sde_logits, k, temperature)
        ode_indices = top_k(ode_logits, k, temperature)
    elif do_top_p:
        sde_indices = top_p(sde_logits, p, temperature)
        ode_indices = top_p(ode_logits, p, temperature)
    else:
        sde_indices = None
        ode_indices = None
        if get_sde:
            sde_indices = argmax_sample(sde_logits)
        if get_ode:
            ode_indices = argmax_sample(ode_logits)

    return sde_solved, ode_solved, sde_indices, ode_indices, sde_path, ode_path

def visualize_embedding_matrix(pl_module):
    embeddings = pl_module.model.encoder.embedding.weight
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


def visualize_latent(pl_module, words_sde, words_ode, tokenizer):
    
    if words_sde is None:
        pass
    else:
        first_latent = words_sde[0] #shape [block_size, hidden_size]
        print(first_latent.shape, "first latent shape")

        for i in range(first_latent.shape[0]):
            latent_vector = first_latent[i]
            print("the latent,", i, " ", latent_vector)
            decoded = pl_module.model.decoder(latent_vector, pl_module.model.encoder.embedding.weight)
            decoded = decoded.argmax(dim=-1).squeeze(-1)
            print(decoded.shape, "decoded shape")
            print(tokenizer.decode([decoded.item()]))

    if words_ode is None:
        pass
    else:
        
        first_latent = words_ode[0]
        print(first_latent.shape, "first latent shape")   

        for i in range(first_latent.shape[0]):
            latent_vector = first_latent[i]
            
            print("the latent,", i, " ", latent_vector)
            decoded = pl_module.model.decoder(latent_vector, pl_module.model.encoder.embedding.weight)
            decoded = decoded.argmax(dim=-1).squeeze(-1)
            print(decoded.shape, "decoded shape")

            print(tokenizer.decode([decoded.item()]))



def visualize_path(pl_module, sde_path, ode_path, logger, tokenizer):
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
    print("sde_path_shape", sde_path.shape)
    if sde_path is None:
        pass
    else:
        for i in range(sde_path.shape[0]):
            print("sde", sde_path[i][0])
            #also print the corersponding word
            decoded = pl_module.model.decoder(sde_path[i][0], pl_module.model.encoder.embedding.weight)
            decoded = decoded.argmax(dim=-1).squeeze(-1)
            print('sde', tokenizer.decode(decoded.tolist()))
    # for the ode path print all the intermediate steps in a row for the first token
    if ode_path is None:
        pass
    else:
        for i in range(ode_path.shape[0]):
            print("ode", ode_path[i][0])
            decoded = pl_module.model.decoder(sde_path[i][0], pl_module.model.encoder.embedding.weight)
            decoded = decoded.argmax(dim=-1).squeeze(-1)
            print("ode", tokenizer.decode(decoded.tolist()))


def idx_to_words(index, tokenizer):
    decoded_texts = []
    for sequence in index:
        decoded_texts.append(tokenizer.decode(sequence.tolist()))
    return decoded_texts
