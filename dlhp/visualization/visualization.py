import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from typing import List, Dict, Tuple
import matplotlib
import torch.nn.functional as F
matplotlib.use('TkAgg')
# -------------------------
# 1) Raster / timeline plot
# -------------------------
def plot_sequence_raster(times: np.ndarray, marks: np.ndarray, id2event: Dict[int,str]=None,
                         title: str = "Event raster", max_events: int = 500,
                         out_dir: str = "./plots", fname: str = "raster.png"):
    os.makedirs(out_dir, exist_ok=True)
    if len(times) == 0:
        return

    if len(times) > max_events:
        idx = np.linspace(0, len(times)-1, max_events).astype(int)
        times = times[idx]
        marks = marks[idx]

    unique_marks = np.unique(marks)
    cmap = plt.get_cmap('tab20')
    plt.figure(figsize=(12, max(2, 0.3*len(unique_marks))))
    for i, m in enumerate(unique_marks):
        mask = (marks == m)
        plt.scatter(times[mask], np.ones(mask.sum())*i, marker='|', s=200,
                    color=cmap(i % 20), label=(id2event[m] if id2event and m in id2event else str(m)))
    plt.yticks(range(len(unique_marks)),
               [id2event[m] if id2event and m in id2event else str(m) for m in unique_marks])
    plt.xlabel("Time")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname))
    plt.close()


# ---------------------------------------
# 2) Cross-correlogram (empirical)
# ---------------------------------------
def cross_correlogram_for_pair(times_i: np.ndarray, times_j: np.ndarray,
                               window: float = 10.0, bin_size: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute histogram of (t_j - t_i) for all pairs with 0 < dt <= window
    Returns (bins_centers, counts_normalized) where counts are normalized per reference event (i)
    """
    nbins = int(np.ceil(window / bin_size))
    bins = np.linspace(0, window, nbins+1)
    counts = np.zeros(nbins, dtype=float)
    # For each event in i, find j-events that occur after it within window
    j_idx = 0
    N_j = len(times_j)
    for t_i in times_i:
        # advance j_idx to first j >= t_i
        while j_idx < N_j and times_j[j_idx] < t_i:
            j_idx += 1
        k = j_idx
        while k < N_j and times_j[k] - t_i <= window:
            dt = times_j[k] - t_i
            bin_idx = min(int(dt // bin_size), nbins-1)
            counts[bin_idx] += 1
            k += 1
    # normalize by number of i-events -> average # of j per i per bin
    if len(times_i) > 0:
        counts = counts / float(len(times_i))
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    return bin_centers, counts

def plot_cross_correlogram_matrix(sequences: List[Dict], event_ids: List[int],
                                  id2event: Dict[int,str]=None,
                                  window: float = 20.0, bin_size: float = 0.5,
                                  out_dir: str = "./plots", fname: str = "correlogram.png"):
    os.makedirs(out_dir, exist_ok=True)
    K = len(event_ids)
    nbins = int(np.ceil(window / bin_size))
    correlograms = np.zeros((K, K, nbins), dtype=float)

    for seq in sequences:
        times = seq['times'].cpu().numpy() if hasattr(seq['times'], 'cpu') else np.array(seq['times'])
        marks = seq['marks'].cpu().numpy() if hasattr(seq['marks'], 'cpu') else np.array(seq['marks'])
        for i_idx, i_id in enumerate(event_ids):
            times_i = times[marks == i_id]
            if len(times_i) == 0:
                continue
            for j_idx, j_id in enumerate(event_ids):
                times_j = times[marks == j_id]
                if len(times_j) == 0:
                    continue
                # compute histogram of lags
                centers, counts = cross_correlogram_for_pair(times_i, times_j, window, bin_size)
                correlograms[i_idx, j_idx, :] += counts

    fig, axes = plt.subplots(K, K, figsize=(3*K, 3*K), squeeze=False)
    for i in range(K):
        for j in range(K):
            ax = axes[i][j]
            ax.bar(centers, correlograms[i,j,:], width=(centers[1]-centers[0]) * 0.9)
            if i==K-1: ax.set_xlabel(f"lag i={event_ids[i]}→j={event_ids[j]}")
            if j==0: ax.set_ylabel("avg # j per i")
    plt.suptitle("Empirical cross-correlograms")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname))
    plt.close()


# -----------------------------------------------------
# 3) Empirical trigger matrix
# -----------------------------------------------------
def plot_trigger_matrix(M: np.ndarray, id2event: Dict[int,str]=None,
                        title: str="Empirical trigger matrix",
                        out_dir: str="./plots", fname: str="trigger_matrix.png"):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8,6))
    plt.imshow(M, interpolation='nearest', cmap='viridis')
    plt.colorbar(label='avg # of triggered j per i (within window)')
    K = M.shape[0]
    labels = [id2event[i] if id2event and i in id2event else str(i) for i in range(K)]
    plt.xticks(range(K), labels, rotation=90)
    plt.yticks(range(K), labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname))
    plt.close()


# ---------------------------------------------------------
# 4) Model-based impulse responses
# ---------------------------------------------------------
def compute_model_impulse_response_dlhp_exact(model, mark_k: int,
                                              t_grid: np.ndarray) -> np.ndarray:
    """
    Compute intensity response curves lambda_j(t) after a single impulse of mark_k at t=0,
    using a trained DLHPExact (stacked LinearDynamicsExact) model.

    Returns array shape (len(t_grid), K) of intensities for each mark j at each t.

    NOTE: this function relies on the model being DLHPExact-like:
      - model.layers[l] must expose .V, .lambda_log_real, .lambda_im (diag-param),
        .E and .alpha (so we can form r_k^{(l)} = E @ alpha[:,k])
      - model.Cs list and final projection W,b,log_s exist.
    """
    device = next(model.parameters()).device
    t_tensor = torch.tensor(t_grid, dtype=torch.float32, device=device)  # (T,)
    L = model.L
    K = model.K

    # For each layer l compute x_l(t) = V_l @ block_diag(exp(lambda_l * t)) @ V_l^{-1} @ r_k^{(l)}
    # We'll compute x_l(t) for all t_grid and stack
    u_prev = torch.zeros((len(t_grid), model.H), dtype=torch.float32, device=device)  # u^{(0)}(t)
    for l in range(L):
        layer = model.layers[l]
        # get r_k for this layer
        # r_k shape (D,) where D = 2*P_l
        r_k = (layer.E @ layer.alpha[:, mark_k]).reshape(-1).to(device)  # (D,)
        P = layer.P
        D = 2 * P
        # V and its inverse
        V = layer.V  # (D,D)
        V_inv = torch.linalg.inv(V) if torch.det(V) != 0 else torch.pinverse(V)
        # get lambda real and imag (P,)
        real = -F.softplus(layer.lambda_log_real)
        imag = layer.lambda_im
        # for each t compute M_exp (2P x 2P) block diag described earlier
        Xl = torch.zeros((len(t_grid), D), dtype=torch.float32, device=device)
        for ti_idx, ti in enumerate(t_tensor):
            # build block-diagonal exp matrix M as in model.expA_dt
            exps = torch.exp(real * float(ti))
            cosbs = torch.cos(imag * float(ti))
            sinbs = torch.sin(imag * float(ti))
            M = torch.zeros(D, D, device=device)
            for i_mode in range(P):
                idx = 2 * i_mode
                a = exps[i_mode] * cosbs[i_mode]
                b = -exps[i_mode] * sinbs[i_mode]
                c = exps[i_mode] * sinbs[i_mode]
                d = exps[i_mode] * cosbs[i_mode]
                M[idx:idx + 2, idx:idx + 2] = torch.tensor([[a, b], [c, d]], device=device)
            Aexp = V @ M @ V_inv
            x_t = Aexp @ r_k  # (D,)
            Xl[ti_idx] = x_t
        # compute y_l(t) = x_l(t) @ C^{(l)} + u_prev(t)
        C = model.Cs[l]  # (D, H)
        # Note: in our DLHPExact C was (2P, H) so x_l @ C gives (T, H)
        y_l = Xl @ C  + u_prev  # (T, H)
        # u_l(t) = LayerNorm(GELU(y_l + u_prev)) but layernorm uses statistics across H -> apply samplewise
        # We'll mimic the model's forward: u = layernorm(act(y) + u)
        # For approximate impulse response we apply same nonlinearity and no learned layernorm affine
        u_l = torch.tensor((torch.nn.functional.gelu(y_l + u_prev).detach().cpu().numpy()), device=device)
        # apply layernorm manually (normalize across H for each time point)
        mean = u_l.mean(dim=1, keepdim=True)
        std = u_l.std(dim=1, keepdim=True) + 1e-6
        u_l = (u_l - mean) / std
        u_prev = u_l  # feed to next layer

    # final projection to intensities
    W = model.W.to(device)   # (K, H)
    b = model.b.to(device)   # (K,)
    s = torch.exp(model.log_s.to(device))
    # u_prev: (T,H)
    proj = (u_prev @ W.t()) + b  # (T,K)
    intensity = s * torch.nn.functional.softplus(proj / s)
    return intensity.detach().cpu().numpy()  # (T,K)


def plot_impulse_responses_from_model(model, id2event: Dict[int,str],
                                      marks_to_plot: List[int]=None,
                                      t_max: float = 20.0, n_steps: int = 200,
                                      out_dir: str="./plots"):
    os.makedirs(out_dir, exist_ok=True)
    if marks_to_plot is None:
        marks_to_plot = list(range(model.K))
    t_grid = np.linspace(0.0, t_max, n_steps)
    for k in marks_to_plot:
        intensity = compute_model_impulse_response_dlhp_exact(model, k, t_grid)  # (T,K)
        plt.figure(figsize=(10, 4))
        for j in range(intensity.shape[1]):
            plt.plot(t_grid, intensity[:, j], label=(id2event[j] if id2event and j in id2event else str(j)))
        plt.title(f"Impulse response after event {k} ({id2event[k] if id2event and k in id2event else ''})")
        plt.xlabel("time after impulse")
        plt.ylabel("intensity")
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"impulse_response_mark{k}.png")
        plt.savefig(out_path)
        plt.close()
