import torch
import numpy as np
import pickle

def run_diagnostic():
    print("Loading validation dataset...")
    # Using the same data path as dataloader
    data_path = 'data/mosi_data.pkl'
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("Data not found. Make sure data/mosi_data.pkl exists.")
        return

    valid_vision = data['valid']['vision']
    
    # Convert to tensor to run Claude's checks
    x_vision = torch.tensor(valid_vision).float()
    
    print(f"Vision feature shape: {x_vision.shape}")
    print(f"Vision feature mean: {x_vision.mean():.4f}")
    print(f"Vision feature std:  {x_vision.std():.4f}")
    print(f"Vision feature min:  {x_vision.min():.4f}")
    print(f"Vision feature max:  {x_vision.max():.4f}")
    print(f"Vision NaN count:    {torch.isnan(x_vision).sum()}")
    print(f"Vision zero rows:    {(x_vision.abs().sum(dim=-1) == 0).sum()}")

if __name__ == '__main__':
    run_diagnostic()
