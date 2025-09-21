"""
Training script for the Deep Linear Hawkes Process (DLHP) model.

This script loads event sequence data from the specified CSV file,
initializes the DLHPExact model, and runs the training loop with validation.
"""

import os
import time
import torch
from dlhp.models.model_exmaple import DLHPExact, train
from dlhp.dataloader.dataloader_example import ErrorLogDLHPDataset, create_data_loaders

def main():
    # 1. Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_PATHS = ["/Users/tr/Desktop/Neural-Hawkes-Process/dlhp/data/ViewECbyOEE_Line1_1.csv"]
    EPOCHS = 25
    BATCH_SIZE = 2  # Use a small batch size as sequences can be long and memory usage can be high
    LR = 1e-3
    MC_SAMPLES = 100  # Number of Monte Carlo samples for integral approximation in log-likelihood
    
    # Early stopping and logging configuration
    PATIENCE = 5
    LOG_DIR = os.path.join("runs", "dlhp_experiment_" + time.strftime("%Y%m%d-%H%M%S"))
    
    # Dataset split configuration
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1

    # Model Hyperparameters
    L = 2      # Number of layers in the DLHP stack
    P = 16     # Number of complex modes per layer (latent dimension will be 2*P)
    H = 64     # Dimension of the residual stream

    # 2. Load data and create data loaders
    print("Loading data...")
    # The dataset processes the CSV and converts it into sequences of events.
    dataset = ErrorLogDLHPDataset(file_paths=DATA_PATHS)
    
    if len(dataset) == 0:
        print("No sequences found in the data. Please check the data file and dataloader.")
        return

    K = len(dataset.event2id)  # Number of unique event types (marks)
    print(f"Found {len(dataset)} sequences.")
    print(f"Found {K} unique event types (marks).")
    
    # Create train, validation, and test data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset,
        batch_size=BATCH_SIZE,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO
    )

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
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        epochs=EPOCHS,
        lr=LR,
        mc_samples=MC_SAMPLES,
        log_dir=LOG_DIR,
        patience=PATIENCE
    )

    # 5. Evaluate on test set
    model.eval()
    total_test_loss = 0.0
    test_count = 0
    print("\nEvaluating on test set...")
    with torch.no_grad():
        for batch in test_loader:
            batch_loss = 0.0
            for item in batch:
                times = item['times'].to(DEVICE)
                marks = item['marks'].to(DEVICE)
                T = float(item['T'])
                if len(times) == 0:
                    continue
                ll = model.log_likelihood(times, marks, T, u_inputs=None, mc_samples=MC_SAMPLES)
                loss = -ll
                batch_loss += loss.item()
            if batch_loss > 0:
                total_test_loss += batch_loss / len(batch)
                test_count += 1
    
    if test_count > 0:
        avg_test_loss = total_test_loss / test_count
        print(f"Test set average loss: {avg_test_loss:.4f}")
    
    print("Training and evaluation finished.")

if __name__ == "__main__":
    main()
