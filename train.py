"""
Training script for the Deep Linear Hawkes Process (DLHP) model.

This script loads event sequence data from the specified CSV file,
initializes the DLHPExact model, and runs the training loop.
"""

import torch
from dlhp.models.model_exmaple import DLHPExact, train
from dlhp.dataloader.dataloader_example import ErrorLogDLHPDataset

def main():
    # 1. Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_PATHS = ["/Users/tr/Desktop/Neural-Hawkes-Process/dlhp/data/ViewECbyOEE_Line1_1.csv"]
    EPOCHS = 25
    BATCH_SIZE = 2  # Use a small batch size as sequences can be long and memory usage can be high
    LR = 1e-3
    MC_SAMPLES = 100  # Number of Monte Carlo samples for integral approximation in log-likelihood

    # Model Hyperparameters
    L = 2      # Number of layers in the DLHP stack
    P = 16     # Number of complex modes per layer (latent dimension will be 2*P)
    H = 64     # Dimension of the residual stream

    # 2. Load data
    print("Loading data...")
    # The dataset processes the CSV and converts it into sequences of events.
    dataset = ErrorLogDLHPDataset(file_paths=DATA_PATHS)
    
    if len(dataset) == 0:
        print("No sequences found in the data. Please check the data file and dataloader.")
        return

    K = len(dataset.event2id)  # Number of unique event types (marks)
    print(f"Found {len(dataset)} sequences.")
    print(f"Found {K} unique event types (marks).")

    # 3. Create model
    print("Creating model...")
    model = DLHPExact(
        L=L, 
        P=P, 
        H=H, 
        K=K, 
        diag_param=True,          # Use the more efficient diagonalizable parameterization
        input_dependent=False     # Assuming dynamics are not input-dependent for now
    )
    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params} trainable parameters.")

    # 4. Start training
    print(f"Starting training on {DEVICE}...")
    # The `train` function is imported from the model file and handles the training loop.
    train(
        model=model, 
        dataset=dataset, 
        device=DEVICE, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        lr=LR, 
        mc_samples=MC_SAMPLES
    )

    print("Training finished.")

if __name__ == "__main__":
    main()
