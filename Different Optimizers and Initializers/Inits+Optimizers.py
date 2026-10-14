# -*- coding: utf-8 -*-

# %% Imports and global configuration

# import random
from pathlib import Path
from itertools import combinations
from time import perf_counter

import torch
from scipy.io import savemat
import numpy as np
import matplotlib.pyplot as plt
import random

torch.set_default_dtype(torch.float64)

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


# ── Problem dimensions ────────────────────────────────────────────────────────

n_workers = 25
dim = 12
subset_size = 23

alpha = torch.eye(dim, dtype=torch.get_default_dtype()).reshape(-1)
subsets = list(combinations(range(n_workers), subset_size))
subset_matrix = torch.tensor(subsets, dtype=torch.long)

resid_ridge = 1e-10
condition_ridge = 1e-6
eps = 1e-6
lam_resid, lam_cond = 1.0, 1e-8
lam_normx, lam_normy = 1e-8, 1e-8


# ── Experiment settings ───────────────────────────────────────────────────────

NUM_STEPS = 50000

# seeds = random.sample(range(1, 101), 10)
# print("Selected seeds:", seeds)

seeds = [84]


# SGD mini-batch subset size
SGD_MINIBATCH_SIZE = 256

# Save final optimized X,Y .mat files for MATLAB recovery error
SAVE_FINAL_MAT = True


# Save plots
SAVE_PLOTS = False


# ── Runtime config ────────────────────────────────────────────────────────────

cuda_subset_batch_size = len(subsets)   # full subset batch on GPU
cpu_subset_batch_size = len(subsets)
enable_cuda_compile = False
cuda_compile_mode = "default"


try:
    import torch._inductor.fx_passes.pad_mm as _pad_mm
    _pad_mm.should_pad_bench = lambda *a, **kw: False
except Exception:
    pass


_runtime_cache = {}
_eval_obj_cache = {}


# %% Config / tensor cache

def get_runtime_config(device=None):
    if device == "cuda":
        return {
            "device": "cuda",
            "dtype": torch.float64,
            "subset_batch_size": cuda_subset_batch_size,
        }

    return {
        "device": device,
        "dtype": torch.float64,
        "subset_batch_size": cpu_subset_batch_size,
    }


def get_runtime_tensors(device, dtype):
    cache_key = (str(device), dtype)

    if cache_key not in _runtime_cache:
        _runtime_cache[cache_key] = {
            "alpha_norm_sq": alpha.square().sum().to(device=device, dtype=dtype),
            "subset_matrix": subset_matrix.to(device=device),
            "eye_subset": torch.eye(subset_size, dtype=dtype, device=device),
        }

    return _runtime_cache[cache_key]


# %% Math helpers

def compute_gram_quantities(X, Y):
    xx = X @ X.T
    yy = Y @ Y.T

    gram_workers = xx * yy
    rhs_workers = (X * Y).sum(dim=1)

    # Separate row norms of X and Y
    x_norms = torch.sqrt(
        torch.clamp(xx.diagonal(), min=0.0)
    )
    y_norms = torch.sqrt(
        torch.clamp(yy.diagonal(), min=0.0)
    )

    return gram_workers, rhs_workers, x_norms, y_norms


def batched_subset_gram(gram_workers, subset_chunk):
    return gram_workers[
        subset_chunk.unsqueeze(-1),
        subset_chunk.unsqueeze(-2),
    ]


def batched_subset_rhs(rhs_workers, subset_chunk):
    return rhs_workers[subset_chunk].unsqueeze(-1)


# %% Objective function

def _make_eval_obj(subset_matrix_device, eye_subset, alpha_norm_sq, batch_size):  # buidls obj func

    num_all_subsets = subset_matrix_device.shape[0]

    def eval_obj(X, Y, subset_indices=None, scale_to_full=False):  # calc loss from current X,Y
        gram_workers, rhs_workers, x_norms, y_norms = (
            compute_gram_quantities(X, Y)
        )

        dtype = gram_workers.dtype
        device = gram_workers.device

        eye_b = eye_subset.unsqueeze(0)

        if subset_indices is None:
            subset_matrix_use = subset_matrix_device
        else:
            subset_matrix_use = subset_matrix_device[subset_indices]

        subset_total = torch.zeros(1, dtype=dtype, device=device)

        for start in range(0, subset_matrix_use.shape[0], batch_size):
            subset_chunk = subset_matrix_use[start:start + batch_size]

            gram = batched_subset_gram(gram_workers, subset_chunk)
            rhs = batched_subset_rhs(rhs_workers, subset_chunk)

            # Residual solve
            resid_chol = torch.linalg.cholesky(gram + 1e-10 * eye_b)
            r = torch.cholesky_solve(rhs, resid_chol)

            linear_term = (rhs.transpose(1, 2) @ r).squeeze(-1).squeeze(-1)
            resid_sq = alpha_norm_sq - linear_term

            # Condition penalty
            stabilized = gram + condition_ridge * eye_b
            chol = torch.linalg.cholesky(stabilized)

            Linv = torch.linalg.solve_triangular(
                chol,
                eye_b.expand(chol.shape[0], -1, -1),
                upper=False,
            )

            trace_inv = Linv.pow(2).sum(dim=(-2, -1))

            subset_total = subset_total + (
                lam_resid * resid_sq + lam_cond * trace_inv
            ).sum()

        if subset_indices is not None and scale_to_full:
            scale = num_all_subsets / subset_matrix_use.shape[0]
            subset_total = subset_total * scale

        x_norm_penalty = (x_norms - 1.0).pow(2).sum()
        y_norm_penalty = (y_norms - 1.0).pow(2).sum()

        total = (
            subset_total
            + lam_normx * x_norm_penalty
            + lam_normy * y_norm_penalty
        )

        return total.squeeze(0)

    return eval_obj


def get_eval_obj_fn(device, dtype, subset_batch_size):
    cache_key = (str(device), dtype, subset_batch_size)

    if cache_key in _eval_obj_cache:
        return _eval_obj_cache[cache_key]

    tensors = get_runtime_tensors(device, dtype)

    eval_obj = _make_eval_obj(
        tensors["subset_matrix"],
        tensors["eye_subset"],
        tensors["alpha_norm_sq"],
        subset_batch_size,
    )

    if str(device) == "cuda" and enable_cuda_compile and hasattr(torch, "compile"):
        fn = torch.compile(eval_obj, mode=cuda_compile_mode)
    else:
        fn = eval_obj

    _eval_obj_cache[cache_key] = fn

    return fn


# %% Initialization

def init_factors(seed=0, device=None):

    rc = get_runtime_config(device)

    generator_device = rc["device"] if rc["device"] == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)

    X = torch.randn(
        (n_workers, dim),
        generator=generator,
        device=rc["device"],
        dtype=rc["dtype"],
        requires_grad=True,
    )

    Y = torch.randn(
        (n_workers, dim),
        generator=generator,
        device=rc["device"],
        dtype=rc["dtype"],
        requires_grad=True,
    )

    return X, Y


def normalize_xy(X0, Y0):

    Xn = X0.detach().clone()
    Yn = Y0.detach().clone()

    with torch.no_grad():
        Xn /= Xn.norm(dim=1, keepdim=True).clamp_min(1e-12)
        Yn /= Yn.norm(dim=1, keepdim=True).clamp_min(1e-12)

    return Xn, Yn


# %% SGD mini-batch subset sampler

class ShuffledSubsetBatchSampler:

    def __init__(self, num_subsets, batch_size, device, seed=0):
        self.num_subsets = num_subsets
        self.batch_size = batch_size

        if str(device) == "cuda":
            self.device = torch.device("cuda")
            self.generator = torch.Generator(device=self.device)
        else:
            self.device = torch.device("cpu")
            self.generator = torch.Generator()

        self.generator.manual_seed(seed)

        self.perm = None
        self.pos = 0
        self._shuffle()

    def _shuffle(self):
        self.perm = torch.randperm(
            self.num_subsets,
            device=self.device,
            generator=self.generator,
        )
        self.pos = 0

    def next_batch(self):
        pieces = []
        remaining = self.batch_size

        while remaining > 0:
            available = self.num_subsets - self.pos

            if available == 0:
                self._shuffle()
                available = self.num_subsets

            take = min(remaining, available)

            pieces.append(self.perm[self.pos:self.pos + take])

            self.pos += take
            remaining -= take

        return torch.cat(pieces, dim=0)


# %% Optimizer selector

def make_optimizer(optimizer_name, params, lr):
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adam":
        return torch.optim.Adam(params, lr=lr)

    # if optimizer_name == "rmsprop":
    #     return torch.optim.RMSprop(params, lr=lr, momentum=0.9)

    # elif optimizer_name == "sgd":
    #     return torch.optim.SGD(
    #         params,
    #         lr=lr,
    #         momentum=0.9,
    #         nesterov=True,
    #     )
    else:
        raise ValueError(f"Unknown optimizer_name: {optimizer_name}")


# %% Time helper
def cumulative_time(step_times):
    return np.cumsum(np.array(step_times, dtype=float))


# %% Optimization loop

def optimize_design(
    num_steps=NUM_STEPS,
    lr=1e-3,
    seed=0,
    device=None,
    verbose=True,
    subset_batch_size=None,
    lr_patience=2000,
    lr_factor=0.5,
    min_lr=1e-7,
    early_stop_patience=10000,
    target_tolerance=1e-10,
    optimizer_name="adam",
    init_X=None,
    init_Y=None,
    sgd_minibatch_size=None,
):
    rc = get_runtime_config(device)

    if subset_batch_size is None:
        subset_batch_size = rc["subset_batch_size"]

    # Full objective function.

    full_eval_obj_fn = get_eval_obj_fn(
        rc["device"],
        rc["dtype"],
        subset_batch_size,
    )

    # Mini-batch objective function.
    train_eval_obj_fn = full_eval_obj_fn

    if init_X is None or init_Y is None:
        X, Y = init_factors(seed=seed, device=device)
    else:
        X = init_X.detach().clone().requires_grad_(True)
        Y = init_Y.detach().clone().requires_grad_(True)

    optimizer = make_optimizer(optimizer_name, [X, Y], lr)

    use_minibatch_sgd = (
        optimizer_name.lower() == "sgd"
        and sgd_minibatch_size is not None
    )

    if use_minibatch_sgd:
        tensors = get_runtime_tensors(rc["device"], rc["dtype"])
        num_subsets = tensors["subset_matrix"].shape[0]

        subset_sampler = ShuffledSubsetBatchSampler(
            num_subsets=num_subsets,
            batch_size=sgd_minibatch_size,
            device=rc["device"],
            seed=seed,
        )
    else:
        subset_sampler = None

    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_factor,
            patience=lr_patience,
            threshold=1e-8,
            threshold_mode="rel",
            min_lr=min_lr,
            verbose=True,
        )
    except TypeError:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_factor,
            patience=lr_patience,
            threshold=1e-8,
            threshold_mode="rel",
            min_lr=min_lr,
        )

    history = []  # full obj values for graph
    step_times = []  # optimization loop times

    is_cuda = device == "cuda"

    best_loss = float("inf")
    best_step = -1
    best_X = None
    best_Y = None
    steps_since_best = 0
# ============================================================
# Optimizer update timing only
#    This time includes:
#       - loss used for optimizer update
#       - backward()
#       - optimizer.step()
#    This time does NOT include full-objective monitoring
# ============================================================
    for step in range(num_steps):
        if is_cuda:
            torch.cuda.synchronize()

        t0 = perf_counter()

        optimizer.zero_grad()

        if use_minibatch_sgd:
            subset_idx = subset_sampler.next_batch()

            loss_for_backward = train_eval_obj_fn(
                X,
                Y,
                subset_indices=subset_idx,
                scale_to_full=True,
            )
        else:
            loss_for_backward = train_eval_obj_fn(X, Y)

        loss_for_backward.backward()
        optimizer.step()

        if is_cuda:
            torch.cuda.synchronize()

        step_time = perf_counter() - t0
        step_times.append(step_time)

        # Full-objective monitoring
        # Monitoring after update.
        with torch.no_grad():
            current_loss = full_eval_obj_fn(X, Y).item()

        do_full_monitor = True

        if do_full_monitor:
            scheduler.step(current_loss)

            if current_loss < best_loss:
                best_loss = current_loss
                best_step = step
                best_X = X.detach().clone()
                best_Y = Y.detach().clone()
                steps_since_best = 0
            else:
                steps_since_best += 1

        history.append(current_loss)
        if verbose and (step == 0 or (step + 1) % 1000 == 0):
            avg = sum(step_times) / len(step_times)
            current_lr = optimizer.param_groups[0]["lr"]

            if use_minibatch_sgd:
                batch_info = f"SGD minibatch={sgd_minibatch_size}"
            else:
                batch_info = "full batch"

            print(
                f"step={step+1:5d}, "
                f"objective={current_loss:.6e}, "
                f"best_objective={best_loss:.6e}, "
                f"lr={current_lr:.3e}, "
                f"{batch_info}, "
                f"step_time={step_time:.3f}s, "
                f"avg_step_time={avg:.3f}s"
            )

        if best_loss < target_tolerance:
            if verbose:
                print(f"\nConverged at step {step + 1}.")
            break

        if steps_since_best >= early_stop_patience:
            if verbose:
                print(f"\nEarly stopping at step {step + 1}: no improvement.")
            break

    if verbose:
        print(f"Best step: {best_step + 1}")
        print(f"Best objective: {best_loss:.15e}")

    return best_X, best_Y, history, step_times, best_loss


# %% Post-processing

@torch.no_grad()
def compute_decoding_coefficients(X, Y, subset_batch_size=None):
    if subset_batch_size is None:
        subset_batch_size = get_runtime_config(
            X.device.type)["subset_batch_size"]

    gram_workers, rhs_workers, _, _ = compute_gram_quantities(X, Y)

    tensors = get_runtime_tensors(gram_workers.device, gram_workers.dtype)
    SM = tensors["subset_matrix"]
    eye = tensors["eye_subset"]

    chunks = []

    for start in range(0, SM.shape[0], subset_batch_size):
        sc = SM[start:start + subset_batch_size]

        gram = batched_subset_gram(gram_workers, sc)
        rhs = batched_subset_rhs(rhs_workers, sc)

        gram_stable = gram + resid_ridge * eye.unsqueeze(0)

        chol = torch.linalg.cholesky(gram_stable)
        r = torch.cholesky_solve(rhs, chol).squeeze(-1)

        chunks.append(r)

    return torch.cat(chunks, dim=0)


@torch.no_grad()
def compute_condition_numbers(X, Y, subset_batch_size=None):
    if subset_batch_size is None:
        subset_batch_size = get_runtime_config(
            X.device.type)["subset_batch_size"]

    gram_workers, _, _, _ = compute_gram_quantities(X, Y)

    tensors = get_runtime_tensors(gram_workers.device, gram_workers.dtype)
    SM = tensors["subset_matrix"]
    eye = tensors["eye_subset"]

    chunks = []

    for start in range(0, SM.shape[0], subset_batch_size):
        sc = SM[start:start + subset_batch_size]

        gram = batched_subset_gram(gram_workers, sc)
        stabilized = gram + condition_ridge * eye.unsqueeze(0)

        eigvals = torch.linalg.eigvalsh(stabilized)

        sigma_min = torch.sqrt(torch.clamp(eigvals[:, 0], min=0.0))
        sigma_max = torch.sqrt(torch.clamp(eigvals[:, -1], min=0.0))

        chunks.append(sigma_max / torch.clamp(sigma_min, min=1e-15))

    cond_tensor = torch.cat(chunks, dim=0)

    return cond_tensor, cond_tensor.max().item(), cond_tensor.mean().item()


# %% MATLAB export

def save_to_mat(
    X,
    Y,
    history,
    step_times=None,
    best_seed=None,
    output_path=None,
    subset_batch_size=None,
    init_type=None,
    optimizer_name=None,
    lr=None,
    best_loss=None,
):
    if output_path is None:
        output_path = Path(
            f"coded_design_{subset_size}_of_{n_workers}_"
            f"..._{lam_cond}_{lam_normx}_{lam_normy}.mat"
        )
    else:
        output_path = Path(output_path)

    if subset_batch_size is None:
        subset_batch_size = get_runtime_config(
            X.device.type)["subset_batch_size"]

    decoding_coeff = compute_decoding_coefficients(
        X,
        Y,
        subset_batch_size=subset_batch_size,
    )

    cond_values, worst_cond, avg_cond = compute_condition_numbers(
        X,
        Y,
        subset_batch_size=subset_batch_size,
    )

    savemat(
        output_path,
        {
            "X": X.cpu().numpy(),
            "Y": Y.cpu().numpy(),
            "history": history,
            "step_times": [] if step_times is None else step_times,
            "best_seed": -1 if best_seed is None else best_seed,
            "alpha": alpha.cpu().numpy(),
            "subset_matrix": subset_matrix.cpu().numpy(),
            "decoding_coeff": decoding_coeff.cpu().numpy(),
            "condition_numbers": cond_values.cpu().numpy(),
            "worst_condition_number": worst_cond,
            "average_condition_number": avg_cond,
            "subset_batch_size": subset_batch_size,
            "condition_ridge": condition_ridge,
            "eps": eps,
            "lam_resid": lam_resid,
            "lam_cond": lam_cond,
            "lam_normx": lam_normx,
            "lam_normy": lam_normy,
            "init_type": "" if init_type is None else init_type,
            "optimizer_name": "" if optimizer_name is None else optimizer_name,
            "lr": np.nan if lr is None else lr,
            "best_loss": np.nan if best_loss is None else best_loss,
        },
    )

    return output_path, worst_cond, avg_cond

# %% Comparison saving


def save_optimizer_comparison(
    all_results,
    output_path="optimizer_comparison_adam_sgd_rmsprop_adagrad.mat",
):
    max_len = max(len(r["history"]) for r in all_results)
    num_runs = len(all_results)

    loss_curves = np.full((num_runs, max_len), np.nan)
    time_curves = np.full((num_runs, max_len), np.nan)

    init_types = []
    optimizer_names = []
    seed_values = []
    lr_values = []
    best_losses = []
    final_losses = []
    total_times = []
    avg_step_times = []
    worst_conds = []
    avg_conds = []
    train_batch_sizes = []

    for i, r in enumerate(all_results):
        hist = np.array(r["history"], dtype=float)
        times = np.array(r["cum_time"], dtype=float)

        loss_curves[i, :len(hist)] = hist
        time_curves[i, :len(times)] = times

        init_types.append(r["init_type"])
        optimizer_names.append(r["optimizer"])
        seed_values.append(r["seed"])
        lr_values.append(r["lr"])
        best_losses.append(r["best_loss"])
        final_losses.append(r["final_loss"])
        total_times.append(r["total_time"])
        avg_step_times.append(r["avg_step_time"])
        worst_conds.append(r["worst_condition_number"])
        avg_conds.append(r["average_condition_number"])
        train_batch_sizes.append(r["train_subset_batch_size"])

    savemat(
        output_path,
        {
            "init_types": np.array(init_types, dtype=object),
            "optimizer_names": np.array(optimizer_names, dtype=object),
            "seed_values": np.array(seed_values),
            "lr_values": np.array(lr_values),
            "best_losses": np.array(best_losses),
            "final_losses": np.array(final_losses),
            "total_times": np.array(total_times),
            "avg_step_times": np.array(avg_step_times),
            "worst_condition_numbers": np.array(worst_conds),
            "average_condition_numbers": np.array(avg_conds),
            "train_subset_batch_sizes": np.array(train_batch_sizes),
            "loss_curves": loss_curves,
            "time_curves": time_curves,
            "n_workers": n_workers,
            "dim": dim,
            "subset_size": subset_size,
            "num_total_subsets": len(subsets),
            "sgd_minibatch_size": SGD_MINIBATCH_SIZE,
            "lam_resid": lam_resid,
            "lam_cond": lam_cond,
            "lam_normx": lam_normx,
            "lam_normy": lam_normy,
        },
    )

    return output_path


# %% Plotting

def _optimizer_label(optimizer_name):
    if optimizer_name == "sgd":
        return f"sgd minibatch {SGD_MINIBATCH_SIZE}"
    return optimizer_name


def plot_loss_vs_step_for_seed(all_results, seed_to_plot, init_type):

    fig, ax = plt.subplots(figsize=(8, 5))

    optimizer_styles = {
        "adam": {
            "label": "Adam",
            "color": "blue",
            "linestyle": "-",
            "marker": "o",
        },
        "sgd": {
            "label": f"Mini-batch SGD {SGD_MINIBATCH_SIZE}",
            "color": "orange",
            "linestyle": "--",
            "marker": "^",
        },
        "rmsprop": {
            "label": "RMSprop",
            "color": "green",
            "linestyle": ":",
            "marker": "s",
        },
    }

    optimizer_order = ["adam"]

    # ---------------------------------------------------------
    # Plot fewer points so dashed/dotted styles remain visible
    # ---------------------------------------------------------
    plot_stride = 150

    for optimizer_name in optimizer_order:
        for r in all_results:
            if (
                r["seed"] == seed_to_plot
                and r["init_type"] == init_type
                and r["optimizer"] == optimizer_name
            ):
                steps = np.arange(1, len(r["history"]) + 1)
                loss = np.array(r["history"])

                # Downsample only for plotting
                steps_plot = steps[::plot_stride]
                loss_plot = loss[::plot_stride]

                # Ensure the very last point is included
                if steps_plot[-1] != steps[-1]:
                    steps_plot = np.append(steps_plot, steps[-1])
                    loss_plot = np.append(loss_plot, loss[-1])

                style = optimizer_styles[optimizer_name]

                # Marker spacing on the already-downsampled curve
                marker_interval = max(1, len(steps_plot) // 12)

                ax.plot(
                    steps_plot,
                    loss_plot,
                    label=style["label"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=2.5,
                    marker=style["marker"],
                    markersize=7,
                    markevery=marker_interval,
                    markerfacecolor="white",
                    markeredgecolor=style["color"],
                    markeredgewidth=1.7,
                )

    ax.set_xlabel("Step", fontsize=15)
    ax.set_ylabel("Objective Loss", fontsize=15)
    ax.set_yscale("log")

    ax.tick_params(axis="both", which="major", labelsize=13)

    matching_results = [
        r for r in all_results
        if r["seed"] == seed_to_plot
        and r["init_type"] == init_type
    ]

    max_steps = max(len(r["history"]) for r in matching_results)

    tick_positions = np.arange(0, max_steps + 1, 10000)
    tick_labels = []

    for tick in tick_positions:
        if tick == 0:
            tick_labels.append("0")
        elif tick == 10000:
            tick_labels.append(r"$10^4$")
        else:
            multiplier = int(tick / 10000)
            tick_labels.append(rf"${multiplier}\times10^4$")

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=13)
    ax.set_xlim(0, max_steps)

    legend = ax.legend(
        loc="upper right",
        fontsize=15,
        handlelength=2.8,
        handletextpad=0.9,
        labelspacing=0.8,
        borderpad=0.75,
        markerscale=1.2,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="black",
    )
    legend.get_frame().set_linewidth(1.2)

    ax.grid(
        True,
        which="major",
        linestyle="-",
        linewidth=0.6,
        alpha=0.45,
    )

    fig.tight_layout()

    output_file = (
        f"objective_loss_vs_step_{init_type}_seed_{seed_to_plot}.png"
    )

    plt.savefig(
        output_file,
        dpi=600,
        bbox_inches="tight",
    )

    plt.show()

    return output_file

# %% Main experiment


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else None
    rc = get_runtime_config(device)

    if device == "cuda":
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        print(
            f"CUDA runtime config: dtype={rc['dtype']}, "
            f"subset_batch_size={rc['subset_batch_size']}, "
            f"compile={enable_cuda_compile}, "
            f"compile_mode={cuda_compile_mode}"
        )

    print(f"Total number of subsets: {len(subsets)}")
    print(f"SGD mini-batch subset size: {SGD_MINIBATCH_SIZE}")

    optimizer_settings = {
        "adam": {
            "lr": 1e-2,
            "sgd_minibatch_size": None,
        }
        # "sgd": {
        #     "lr": 1e-3,
        #     "sgd_minibatch_size": SGD_MINIBATCH_SIZE,
        # },
        # "rmsprop": {
        #     "lr": 1e-3,
        #     "sgd_minibatch_size": None,
        # },
        # "adagrad": {
        #     "lr": 1e-2,
        #     "sgd_minibatch_size": None,
        # },
    }

    all_results = []

    for seed in seeds:
        print("\n======================================")
        print(f"Generating ONE base raw X0,Y0 for seed {seed}")
        print("======================================")

        # One base raw initialization for this seed
        base_X0, base_Y0 = init_factors(seed=seed, device=device)

        base_X0 = base_X0.detach()
        base_Y0 = base_Y0.detach()

        # Raw version
        raw_X0 = base_X0.detach().clone()
        raw_Y0 = base_Y0.detach().clone()

        # Normalized version from the SAME raw base
        # norm_X0, norm_Y0 = normalize_xy(base_X0, base_Y0)

        # MatDot-style initialization for this seed
        # matdot_X0, matdot_Y0 = init_factors_matdot(seed=seed, device=device)
        # matdot_X0 = matdot_X0.detach()
        # matdot_Y0 = matdot_Y0.detach()

        init_dict = {
            "raw": (raw_X0, raw_Y0),
            # "normalized": (norm_X0, norm_Y0),
            # "matdot": (matdot_X0, matdot_Y0),
        }

        for init_type, (X0, Y0) in init_dict.items():
            print("\n######################################")
            print(f"Initialization type: {init_type}, seed: {seed}")
            print("######################################")

            for optimizer_name, cfg in optimizer_settings.items():
                print("\n--------------------------------------")
                print(f"Initialization: {init_type}")
                print(f"Optimizer: {optimizer_name}, Seed: {seed}")

                if optimizer_name == "sgd":
                    print(
                        f"Training subset batch size: {SGD_MINIBATCH_SIZE} with shuffle")
                else:
                    print(
                        f"Training subset batch size: full batch = {len(subsets)}")

                print("--------------------------------------")

                X, Y, history, step_times, best_loss = optimize_design(
                    num_steps=NUM_STEPS,
                    lr=cfg["lr"],
                    seed=seed,
                    device=device,
                    verbose=True,
                    subset_batch_size=rc["subset_batch_size"],
                    optimizer_name=optimizer_name,
                    init_X=X0,
                    init_Y=Y0,
                    sgd_minibatch_size=cfg["sgd_minibatch_size"],
                )

                cond_values, worst_cond, avg_cond = compute_condition_numbers(
                    X,
                    Y,
                    subset_batch_size=rc["subset_batch_size"],
                )

                if SAVE_FINAL_MAT:
                    final_output_path = Path(
                        f"optimized_{init_type}_{optimizer_name}_seed_{seed}_"
                        f"{subset_size}_of_{n_workers}_"
                        f"torch_batched_result_{lam_cond}_{lam_normx}_{lam_normy}.mat"
                    )

                    save_to_mat(
                        X,
                        Y,
                        history,
                        step_times=step_times,
                        best_seed=seed,
                        output_path=final_output_path,
                        subset_batch_size=rc["subset_batch_size"],
                        init_type=init_type,
                        optimizer_name=optimizer_name,
                        lr=cfg["lr"],
                        best_loss=best_loss,
                    )

                    print("Saved optimized X,Y MAT file:", final_output_path)

                cum_time = cumulative_time(step_times)
                total_time = float(cum_time[-1])
                avg_step_time = float(total_time / len(step_times))

                if optimizer_name == "sgd":
                    train_subset_batch_size = SGD_MINIBATCH_SIZE
                else:
                    train_subset_batch_size = len(subsets)

                result = {
                    "init_type": init_type,
                    "seed": seed,
                    "optimizer": optimizer_name,
                    "lr": cfg["lr"],
                    "history": history,
                    "step_times": step_times,
                    "cum_time": cum_time,
                    "best_loss": best_loss,
                    "final_loss": history[-1],
                    "total_time": total_time,
                    "avg_step_time": avg_step_time,
                    "worst_condition_number": worst_cond,
                    "average_condition_number": avg_cond,
                    "train_subset_batch_size": train_subset_batch_size,
                }

                all_results.append(result)

                print(
                    f"\nResult for init={init_type}, optimizer={optimizer_name}, seed={seed}")
                print(f"Best loss:                 {best_loss:.6e}")
                print(f"Final loss:                {history[-1]:.6e}")
                print(f"Total time:                {total_time:.3f} s")
                print(f"Average step time:         {avg_step_time:.6f} s")
                print(f"Worst condition number:    {worst_cond:.6e}")
                print(f"Average condition number:  {avg_cond:.6e}")
                print(f"Training subset batch size:{train_subset_batch_size}")

    if SAVE_PLOTS:
        for init_type in ["normalized"]:

            for s in seeds:

                step_plot_file = plot_loss_vs_step_for_seed(
                    all_results,
                    seed_to_plot=s,
                    init_type=init_type,
                )

        print(
            "Saved objective loss vs step plot:",
            step_plot_file
        )
