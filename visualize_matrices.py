import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from model import GPTConfig, GPT  # Ensure model.py is accessible

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize model weights')
    parser.add_argument('--ckpt_iter', type=int, default=10000)
    parser.add_argument('--config', type=str, default='1_1_120')
    parser.add_argument('--device', type=str, default='cpu')  # Using CPU for visualization is usually fine
    parser.add_argument('--num_nodes', type=int, default=100)
    parser.add_argument('--num_of_paths', type=int, default=20)
    parser.add_argument('--dataset', type=str, default='simple_graph')
    return parser.parse_args()

def visualize_weights(model, save_dir):
    """
    Extract and visualize FFN (W1, W2), compute W^M = W2 * W1, and extract Value matrix W_V.
    """
    os.makedirs(save_dir, exist_ok=True)  # Ensure output directory exists

    for i, block in enumerate(model.transformer.h):  # Loop through transformer blocks
        # --- Extract FFN weight matrices ---
        ffn_w1 = block.mlp.c_fc.weight.data.cpu().numpy()  # First FFN matrix (input → higher dimension)
        ffn_w2 = block.mlp.c_proj.weight.data.cpu().numpy()  # Second FFN matrix (higher dimension → output)

        # Compute W^M = W2 * W1 (corrected order)
        wm = np.dot(ffn_w2, ffn_w1)

        # --- Extract Value matrix (last third of c_attn weights) ---
        value_matrix = block.attn.c_attn.weight.data.cpu().numpy()  # Full attention weight matrix
        n_embd = value_matrix.shape[1]  # Embedding dimension
        value_matrix = value_matrix[:, -n_embd:]  # Extract only the Value matrix part

        # Save W^M and Value matrix as numpy files
        np.save(os.path.join(save_dir, f'layer_{i}_WM.npy'), wm)
        np.save(os.path.join(save_dir, f'layer_{i}_Value.npy'), value_matrix)

        # --- Visualization ---
        plt.figure(figsize=(10, 8))
        plt.imshow(ffn_w1, cmap='viridis', aspect='auto')
        plt.colorbar(label='Weight Value')
        plt.title(f'Layer {i} - FFN W1 Weights')
        plt.xlabel('Input Feature')
        plt.ylabel('Output Feature')
        plt.savefig(os.path.join(save_dir, f'layer_{i}_ffn_w1.png'), dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 8))
        plt.imshow(ffn_w2, cmap='viridis', aspect='auto')
        plt.colorbar(label='Weight Value')
        plt.title(f'Layer {i} - FFN W2 Weights')
        plt.xlabel('Input Feature')
        plt.ylabel('Output Feature')
        plt.savefig(os.path.join(save_dir, f'layer_{i}_ffn_w2.png'), dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 8))
        plt.imshow(wm, cmap='viridis', aspect='auto')
        plt.colorbar(label='Weight Value')
        plt.title(f'Layer {i} - W^M (W2 * W1)')
        plt.xlabel('Input Feature')
        plt.ylabel('Output Feature')
        plt.savefig(os.path.join(save_dir, f'layer_{i}_WM.png'), dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 8))
        plt.imshow(value_matrix, cmap='viridis', aspect='auto')
        plt.colorbar(label='Weight Value')
        plt.title(f'Layer {i} - Value Matrix Weights')
        plt.xlabel('Input Feature')
        plt.ylabel('Output Feature')
        plt.savefig(os.path.join(save_dir, f'layer_{i}_Value.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    args = parse_args()
    
    dataset = args.dataset
    ckpt_iter = args.ckpt_iter
    device = args.device
    num_nodes = args.num_nodes
    num_of_paths = args.num_of_paths
    config = args.config

    out_dir = f'out/{dataset}_{config}_{num_nodes}/'  # Output directory

    # Determine checkpoint path
    if num_of_paths == 0:
        ckpt_path = os.path.join(out_dir, f'{ckpt_iter}_ckpt.pt')
    else:
        ckpt_path = os.path.join(out_dir, f'{ckpt_iter}_ckpt_{num_of_paths}.pt')

    print(f"Loading checkpoint from {ckpt_path}...")

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Initialize GPT model with checkpoint config
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)

    # Load model state dict (fix unwanted prefixes if needed)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    # Define visualization directory
    vis_dir = os.path.join(out_dir, f'weight_visualizations_{ckpt_iter}')

    # Visualize weights
    print(f"Visualizing weights and saving to {vis_dir}...")
    visualize_weights(model, vis_dir)
    print("Visualization complete!")

if __name__ == "__main__":
    main()
