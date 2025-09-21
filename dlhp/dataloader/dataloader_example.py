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
from torch.utils.data import Dataset

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
