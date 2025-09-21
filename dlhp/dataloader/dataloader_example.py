import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

class ErrorLogDLHPDataset(Dataset):
    def __init__(self, file_paths, time_col='TimeStamp', st_cols=None, oee_col='OEECause', oee_st_col=None):
        """
        Args:
            file_paths: list of CSV file paths
            time_col: name of the timestamp column
            st_cols: list of ST error columns (default ST1_Err ... ST4_Err)
            oee_col: name of the OEE cause column
            oee_st_col: name of the column containing ST information (e.g., 'OEE_ST')
        """
        self.file_paths = file_paths
        self.time_col = time_col
        self.oee_col = oee_col
        self.oee_st_col = oee_st_col

        # load all dataframes
        dfs = [pd.read_csv(fp) for fp in file_paths]
        df = pd.concat(dfs, ignore_index=True)

        # Verify required columns exist
        if time_col not in df.columns:
            raise ValueError(f"TimeStamp column '{time_col}' not found in the data")
        if oee_col not in df.columns:
            raise ValueError(f"OEE cause column '{oee_col}' not found in the data")

        # Get ST column based on OEE_st if provided
        if self.oee_st_col is not None:
            if self.oee_st_col not in df.columns:
                raise ValueError(f"OEE_ST column '{self.oee_st_col}' not found in the data")
            
            # Get the ST number from OEE_st column
            st_values = df[self.oee_st_col].dropna().unique().tolist()
            if not st_values:
                raise ValueError(f"No ST values found in {self.oee_st_col}")
            
            st_number = st_values[0]  # Take the first ST value
            st_col = f"{st_number}_Err"
            
            if st_col not in df.columns:
                raise ValueError(f"Column {st_col} not found in the data")
            
            self.st_cols = [st_col]
            # print(f"Using ST column based on {self.oee_st_col}: {st_col}")
        else:
            # Original dynamic ST columns detection
            if st_cols is None:
                self.st_cols = [col for col in df.columns if col.startswith('ST') and col.endswith('_Err')]
                if not self.st_cols:
                    raise ValueError("No ST*_Err columns found in the data")
                # print(f"Found ST columns: {', '.join(self.st_cols)}")
            else:
                # Verify that provided st_cols exist in the data
                missing_cols = [col for col in st_cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Following ST columns were not found in the data: {', '.join(missing_cols)}")
                self.st_cols = st_cols

        # build vocab of all event types (ErrorCodes + OEECause)
        st_values = []
        for col in self.st_cols:
            st_values.extend(df[col].dropna().unique().tolist())
        oee_values = df[self.oee_col].dropna().unique().tolist()

        # assign unique ids
        self.event2id = {}
        st_values = sorted(set(st_values))  # 确保唯一性和确定性顺序
        oee_values = sorted(set(oee_values))  # 确保唯一性和确定性顺序

        # print("Found ST events:", st_values)
        # print("Found OEE events:", oee_values)

        # 首先分配 ST 事件的 ID
        idx = 0
        for v in st_values:
            if pd.notna(v) and str(v).strip():  # 确保值有效
                key = f"ST:{v}"
                if key not in self.event2id:
                    self.event2id[key] = idx
                    idx += 1

        # 然后分配 OEE 事件的 ID
        for v in oee_values:
            if pd.notna(v) and str(v).strip():  # 确保值有效
                key = f"OEE:{v}"
                if key not in self.event2id:
                    self.event2id[key] = idx
                    idx += 1

        self.id2event = {i: e for e, i in self.event2id.items()}

        # print("Event ID mapping:")
        # for event, idx in sorted(self.event2id.items(), key=lambda x: x[1]):
        #     print(f"  {idx}: {event}")

        # convert df to list of sequences grouped by day or file (simple choice)
        # Here we treat each file as one sequence
        self.sequences = []
        for fp, df_seq in zip(file_paths, dfs):
            events = []
            for _, row in df_seq.iterrows():
                try:
                    t = pd.to_datetime(row[self.time_col]).timestamp()
                except (ValueError, TypeError) as e:
                    print(f"Warning: Invalid timestamp in row {row.name}: {row[self.time_col]}")
                    continue

                # Process ST error columns
                for col in self.st_cols:
                    try:
                        val = row[col]
                        if pd.notna(val) and val != '':  # Check for both NaN and empty string
                            event_key = f"ST:{val}"
                            if event_key in self.event2id:  # Only add if we have seen this event type before
                                events.append((t, self.event2id[event_key]))
                            else:
                                print(f"Warning: Unknown ST event '{val}' in column {col}")
                    except KeyError:
                        # This shouldn't happen now due to our column validation, but keep as safety
                        print(f"Warning: Column {col} not found in data")
                        continue

                # Process OEE cause
                try:
                    val = row[self.oee_col]
                    if pd.notna(val) and val != '':
                        event_key = f"OEE:{val}"
                        if event_key in self.event2id:
                            events.append((t, self.event2id[event_key]))
                        else:
                            print(f"Warning: Unknown OEE event '{val}'")
                except KeyError:
                    print(f"Warning: OEE column {self.oee_col} not found in data")

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