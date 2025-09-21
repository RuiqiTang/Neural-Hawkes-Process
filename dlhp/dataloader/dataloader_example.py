"""
DLHP DataLoader for tabular error log data

This module implements a PyTorch Dataset and collate function that
transform tabular machine error logs with columns:
    TimeStamp, ST1_Err, ST2_Err, ST3_Err, ST4_Err, OEECause
into event sequences suitable for DLHP training.

Each row may contain multiple event triggers (for each STx_Err and OEE cause).
We map each possible ErrorCode and OEECause to unique integer marks.
The Dataset then outputs sequences with fields: times, marks, T.

"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

class ErrorLogDLHPDataset(Dataset):
    def __init__(self, file_paths, time_col='TimeStamp', st_cols=None, oee_col='OEECause'):
        """
        Args:
            file_paths: list of CSV file paths
            time_col: name of the timestamp column
            st_cols: list of ST error columns (default ST1_Err ... ST4_Err)
            oee_col: name of the OEE cause column
        """
        self.file_paths = file_paths
        self.time_col = time_col
        self.oee_col = oee_col

        # load all dataframes
        dfs = [pd.read_csv(fp) for fp in file_paths]
        df = pd.concat(dfs, ignore_index=True)

        # Dynamically find st_cols from dataframe columns
        if st_cols is None:
            self.st_cols = [col for col in df.columns if col.startswith('ST') and col.endswith('_Err')]
        else:
            self.st_cols = st_cols

        # build vocab of all event types (ErrorCodes + OEECause)
        st_values = []
        for col in self.st_cols:
            st_values.extend(df[col].dropna().unique().tolist())
        oee_values = df[self.oee_col].dropna().unique().tolist()

        # assign unique ids
        self.event2id = {}
        idx = 0
        for v in st_values:
            if v not in self.event2id:
                self.event2id[f"ST:{v}"] = idx
                idx += 1
        for v in oee_values:
            if v not in self.event2id:
                self.event2id[f"OEE:{v}"] = idx
                idx += 1
        self.id2event = {i: e for e, i in self.event2id.items()}

        # convert df to list of sequences grouped by day or file (simple choice)
        # Here we treat each file as one sequence
        self.sequences = []
        for fp, df_seq in zip(file_paths, dfs):
            events = []
            for _, row in df_seq.iterrows():
                t = pd.to_datetime(row[self.time_col]).timestamp()
                for col in self.st_cols:
                    val = row[col]
                    if pd.notna(val):
                        events.append((t, self.event2id[f"ST:{val}"]))
                val = row[self.oee_col]
                if pd.notna(val):
                    events.append((t, self.event2id[f"OEE:{val}"]))
            # sort events by time
            events.sort(key=lambda x: x[0])
            if not events:
                continue
            times = torch.tensor([e[0] for e in events], dtype=torch.float32)
            marks = torch.tensor([e[1] for e in events], dtype=torch.long)
            T = float(times.max())
            self.sequences.append({'times': times, 'marks': marks, 'T': T})

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def collate_variable(batch):
    # Keep batch as list of dicts for DLHP training loop
    return batch

def create_data_loaders(dataset, batch_size=8, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Create train, validation, and test dataloaders from a dataset.
    
    Args:
        dataset: The full dataset
        batch_size: Batch size for the dataloaders
        train_ratio: Proportion of data to use for training
        val_ratio: Proportion of data to use for validation
        test_ratio: Proportion of data to use for testing
        seed: Random seed for reproducibility
        
    Returns:
        train_loader, val_loader, test_loader
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), \
        "Ratios must sum to 1"
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Calculate lengths for splits
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size  # Take the rest to ensure we use all data
    
    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_variable
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_variable
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_variable
    )
    
    print(f"Dataset split sizes:")
    print(f"Train: {len(train_dataset)} sequences")
    print(f"Validation: {len(val_dataset)} sequences")
    print(f"Test: {len(test_dataset)} sequences")
    
    return train_loader, val_loader, test_loader
