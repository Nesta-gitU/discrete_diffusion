import torch
import wandb 
import matplotlib.pyplot as plt
import numpy as np

from src.utils.differential_equations import sde_drift, solve_de



def sample_code(pl_module, datamodule, logger, get_sde, get_ode, n_steps, batch_size): #add clamping as an option, also add sampling instead of argmax as an option later, 
        tokenizer = datamodule.tokenizer
        text_table = wandb.Table(columns=["epoch", "global_step", "text_ode", "text_sde"])

        words_sde, words_ode, sde_path, ode_path = sample_from_diffusion(pl_module, batch_size, get_sde, get_ode, n_steps)

        #visualize the embedding matrix with color coding 
        print("visualizing embedding matrix")
        visualize_embedding_matrix(pl_module)

        #visualize the path somehow
        print("visualizing path")
        visualize_path(pl_module, sde_path, ode_path, logger, tokenizer)

        #take the first latent matrix in the batch, print each time the word vector and the word to which it was decoded
        # #maybe also the original word vector
        print("visualizing latent")
        visualize_latent(pl_module, words_sde, words_ode, tokenizer) 

        tokenizer = datamodule.tokenizer

        if words_sde is None:
            w_sde = None
        else:
            w_sde = idx_to_words(words_sde, tokenizer)
        if words_ode is None:
            w_ode = None
        else:
            w_ode = idx_to_words(words_ode, tokenizer)

        text_table.add_data(str(w_ode), str(w_sde))

        logger.experiment.log({"generated_samples": text_table})

def sample_from_diffusion(module, batch_size, get_sde=True, get_ode=True, _n_steps=100):
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
        sde_solved, sde_path = solve_de(z, 1, 0, _n_steps, module, 'sde')
        #decode the sde solve back into words 
        sde_logits = module.model.decoder(sde_solved, module.model.encoder.embedding.weight)
        sde_indices = sde_logits.argmax(dim=-1).squeeze(-1)
    if get_ode:
        ode_solved, ode_path = solve_de(z, 1, 0, _n_steps, module, 'ode')
        #decode the ode solve back into words
        ode_logits = module.model.decoder(ode_solved, module.model.encoder.embedding.weight)
        ode_indices = ode_logits.argmax(dim=-1).squeeze(-1)


    return sde_indices, ode_indices, sde_path, ode_path

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
    plt.show()


def visualize_latent(pl_module, words_sde, words_ode, tokenizer):
    
    if words_ode is None:
        pass
    else:
        first_latent = words_sde #shape [block_size, hidden_size]
        print(first_latent.shape)

        for i in range(first_latent.shape[0]):
            latent_vector = first_latent[i]
            print("the latent,", i, " ", latent_vector)
            decoded = pl_module.model.decoder(first_latent, pl_module.model.encoder.embedding.weight)
            decoded = decoded.argmax(dim=-1).squeeze(-1)[i]

            print(tokenizer.decode(decoded.tolist()))

    if words_sde is None:
        pass
    else:
        first_latent = words_ode[0]

        for i in range(first_latent.shape[0]):
            latent_vector = first_latent[i]
            print("the latent,", i, " ", latent_vector)
            decoded = pl_module.model.decoder(first_latent, pl_module.model.encoder.embedding.weight)
            decoded = decoded.argmax(dim=-1).squeeze(-1)[i]

            print(tokenizer.decode(decoded.tolist()))



def visualize_path(pl_module, sde_path, ode_path, logger, tokenizer):
    # for the sde path print all the intermediate steps in a row for the first token
    print("sde path")
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
    if sde_path is None:
        pass
    else:
        for i in range(sde_path.shape[1]):
            print("sde", sde_path[0][i])
            #also print the corersponding word
            decoded = pl_module.model.decoder(sde_path[0][i], pl_module.model.encoder.embedding.weight)
            decoded = decoded.argmax(dim=-1).squeeze(-1)
            print('sde', tokenizer.decode(decoded.tolist()))
    # for the ode path print all the intermediate steps in a row for the first token
    if ode_path is None:
        pass
    else:
        for i in range(ode_path.shape[1]):
            print("ode", ode_path[0][i])
            decoded = pl_module.model.decoder(sde_path[0][i], pl_module.model.encoder.embedding.weight)
            decoded = decoded.argmax(dim=-1).squeeze(-1)
            print("ode", tokenizer.decode(decoded.tolist()))


def idx_to_words(index, tokenizer):
    decoded_texts = []
    for sequence in index:
        decoded_texts.append(tokenizer.decode(sequence.tolist()))
    return decoded_texts
