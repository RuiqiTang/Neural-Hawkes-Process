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
import glob
import importlib.util
from dlhp.visualization.visualization import *

def main():
    # 1. Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pattern=r"G:\CodeRemote\Neural-Hawkes-Process\dlhp\data\TrainingSet\*.csv"
    DATA_PATHS = glob.glob(pattern)
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
    dataset = ErrorLogDLHPDataset(
        file_paths=['your_file.csv'],
        time_col='TimeStamp',
        oee_col='OEECause',
        oee_st_col='OEE_ST'  # 这里指定包含ST信息的列
    )
    
    if len(dataset) == 0:
        print("No sequences found in the data. Please check the data file and dataloader.")
        return

    K = len(dataset.event2id)  # Number of unique event types (marks)
    print(f"Found {len(dataset)} sequences.")
    print(f"Found {K} unique event types (marks).")
    
    # Verify marks are within valid range
    max_mark = -1
    for seq in dataset.sequences:
        if len(seq['marks']) > 0:
            max_mark = max(max_mark, seq['marks'].max().item())
    
    if max_mark >= K:
        raise ValueError(f"Found mark ID {max_mark} which is >= number of event types {K}. "
                        f"This indicates a mismatch between event_id mapping and the marks in the data.")
    
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

    # 6. Visualization
    print("\nGenerating visualizations...")
    
    # Create visualization directory
    viz_dir = os.path.join(LOG_DIR, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Get all sequences from test set for visualization
    test_sequences = []
    for batch in test_loader:
        test_sequences.extend(batch)

    # Select first sequence for raster plot
    sample_seq = test_sequences[0]
    sample_times = sample_seq['times'].cpu().numpy()
    sample_marks = sample_seq['marks'].cpu().numpy()

    # 1. Plot event sequence raster
    print("Plotting event sequence raster...")
    plot_sequence_raster(
        times=sample_times,
        marks=sample_marks,
        id2event=dataset.id2event,
        title="Sample Event Sequence",
        max_events=500,
        out_dir=viz_dir,
        fname="event_sequence_raster.png"
    )

    # 2. Plot cross-correlogram matrix
    print("Plotting cross-correlogram matrix...")
    # Get all unique event IDs in sorted order
    event_ids = sorted(dataset.event2id.values())
    plot_cross_correlogram_matrix(
        sequences=test_sequences,  # Use collected test sequences
        event_ids=event_ids,
        id2event=dataset.id2event,
        window=20.0,  # 20秒的时间窗口
        bin_size=0.5,  # 0.5秒的bin大小
        out_dir=viz_dir,
        fname="cross_correlogram_matrix.png"
    )

    # 3. Generate trigger matrix for whole test set
    print("Generating trigger matrix...")
    # 计算trigger matrix (使用相同的window参数保持一致性)
    trigger_matrix = np.zeros((len(event_ids), len(event_ids)))
    for seq in test_sequences:
        times = seq['times'].cpu().numpy()
        marks = seq['marks'].cpu().numpy()
        for i, mark_i in enumerate(event_ids):
            mask_i = (marks == mark_i)
            if not mask_i.any():
                continue
            times_i = times[mask_i]
            for j, mark_j in enumerate(event_ids):
                mask_j = (marks == mark_j)
                if not mask_j.any():
                    continue
                times_j = times[mask_j]
                # Count events of type j within window after each event of type i
                for t_i in times_i:
                    trigger_matrix[i, j] += np.sum((times_j > t_i) & (times_j <= t_i + 20.0))

    # Normalize by number of source events
    for i, mark_i in enumerate(event_ids):
        count_i = sum(1 for seq in test_sequences for m in seq['marks'] if m == mark_i)
        if count_i > 0:
            trigger_matrix[i, :] /= count_i

    plot_trigger_matrix(
        M=trigger_matrix,
        id2event=dataset.id2event,
        title="Empirical Trigger Matrix (20s window)",
        out_dir=viz_dir,
        fname="trigger_matrix.png"
    )

    # 4. Plot model's impulse responses
    print("Plotting model impulse responses...")
    model = model.to('cpu')  # Move model to CPU for visualization
    plot_impulse_responses_from_model(
        model=model,
        id2event=dataset.id2event,
        marks_to_plot=event_ids,  # Plot for all event types in sorted order
        t_max=20.0,  # 与trigger matrix使用相同的时间窗口
        n_steps=200,
        out_dir=viz_dir
    )

    print(f"\nVisualizations have been saved to: {viz_dir}")
    print("Training, evaluation and visualization finished.")

if __name__ == "__main__":
    main()
