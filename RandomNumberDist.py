import math
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def generate_random_number(start, end):
    """Generates a random number between start and end (inclusive)."""
    if start > end:
        raise ValueError("Start must be less than or equal to end.")
    return random.randint(start, end)


def generate_random_list(start, end, size):
    """Generates a list of random numbers between start and end (inclusive)."""
    if size < 0:
        raise ValueError("Size must be a non-negative integer.")
    return [generate_random_number(start, end) for _ in range(size)]


def generate_classical_distribution(start, end, size):
    """Generates a classical distribution of random numbers and returns it as a Series."""
    random_numbers = generate_random_list(start, end, size)
    distribution = pd.Series(random_numbers).value_counts().sort_index()
    return distribution


# ── Quantum circuit (built and transpiled once) ───────────────────────────────

def _prepare_circuit(n_qubits: int, simulator: AerSimulator) -> QuantumCircuit:
    """
    Build and transpile a Hadamard-measure circuit once.
    Transpilation costs ~120 ms; doing it once instead of per-batch is the
    single biggest speed win.
    """
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))           # Hadamard -> equal superposition of all 2^n states
    qc.measure(range(n_qubits), range(n_qubits))
    return transpile(qc, simulator)  # compile once, reuse for every run


# ── Core generator ────────────────────────────────────────────────────────────

def generate_quantum_list(start: int, end: int, size: int) -> np.ndarray:
    """
    Return a numpy array of `size` quantum-random integers in [start, end].

    Algorithm
    ---------
    * n_qubits = ceil(log2(range_size))  ->  2^n >= range_size
    * Run the Hadamard circuit for `shots` measurements in one call.
      Shots = ceil(size / acceptance_rate * 1.25) -- sized to satisfy the
      request in a single simulator run ~99% of the time.
    * Expand the histogram (counts dict) with np.repeat, then apply
      vectorised rejection sampling to remove values above `limit`
      that would cause modulo bias.
    * Loop only if the first run fell short (rare).
    """
    if start > end:
        raise ValueError("start must be <= end")
    if size <= 0:
        return np.empty(0, dtype=np.int32)

    range_size = end - start + 1
    n_qubits   = max(1, math.ceil(math.log2(range_size)))
    max_val    = 2 ** n_qubits                        # total circuit outcomes
    limit      = max_val - (max_val % range_size)     # rejection threshold
    acceptance = limit / max_val                      # fraction of shots kept

    simulator = AerSimulator()
    compiled  = _prepare_circuit(n_qubits, simulator)

    output = np.empty(size, dtype=np.int32)
    filled = 0

    while filled < size:
        needed = size - filled
        # 1.25x overhead almost always clears in one pass; cap at 100k shots
        shots  = min(int(math.ceil(needed / acceptance * 1.25)), 100_000)

        counts = simulator.run(compiled, shots=shots).result().get_counts()

        # Vectorised histogram expansion (much faster than Python loops)
        keys = np.fromiter((int(bs, 2) for bs in counts),
                           dtype=np.int32, count=len(counts))
        vals = np.fromiter(counts.values(),
                           dtype=np.int32, count=len(counts))
        raw      = np.repeat(keys, vals)               # expand histogram -> flat array
        accepted = raw[raw < limit] % range_size + start  # rejection + range shift

        # Shuffle before slicing: np.repeat preserves key order, so without
        # this, accepted[:take] systematically over-samples whichever values
        # appeared first in the counts dict, skewing the distribution.
        np.random.default_rng().shuffle(accepted)

        take = min(len(accepted), needed)
        output[filled:filled + take] = accepted[:take]
        filled += take

    return output


def generate_distribution(start: int, end: int, size: int) -> pd.Series:
    """Generate `size` quantum random integers in [start, end] and return counts."""
    print(f"Running Qiskit circuit  ->  {size} random integers in [{start}, {end}] ...")
    numbers = generate_quantum_list(start, end, size)
    return pd.Series(numbers).value_counts().sort_index()


# ── Plotting helpers ──────────────────────────────────────────────────────────

def five_number_summary(dist: pd.Series, size: int) -> dict:
    """Return the five-number summary of the *bin counts* in dist."""
    v = dist.values
    # Pearson r between observed counts and bin index (1..N).
    # Tests for any systematic trend across the range (e.g. low numbers
    # appearing more than high numbers). For a truly uniform distribution
    # r should be close to 0.
    bin_indices = np.arange(len(v), dtype=float)
    r = float(np.corrcoef(v, bin_indices)[0, 1])
    return {
        "Min":    int(v.min()),
        "Q1":     float(np.percentile(v, 25)),
        "Median": float(np.median(v)),
        "Q3":     float(np.percentile(v, 75)),
        "Max":    int(v.max()),
        "Mean":   float(v.mean()),
        "Std":    float(v.std()),
        "r":      r,
    }


def add_summary_box(ax, summary: dict, expected: float, loc: str = "upper right"):
    """Overlay a five-number summary text box on ax."""
    r_val = summary.get("r", float("nan"))
    lines = [
        "Five-Number Summary (bin counts)",
        f"  Min    {summary['Min']}",
        f"  Q1     {summary['Q1']:.1f}",
        f"  Median {summary['Median']:.1f}",
        f"  Q3     {summary['Q3']:.1f}",
        f"  Max    {summary['Max']}",
        f"  Mean   {summary['Mean']:.1f}  (expected {expected:.0f})",
        f"  Std    {summary['Std']:.1f}",
        f"  r      {r_val:+.4f}  (counts vs index)",
    ]
    text = "\n".join(lines)
    x = 0.99 if "right" in loc else 0.01
    ha = "right" if "right" in loc else "left"
    y = 0.97 if "upper" in loc else 0.03
    va = "top" if "upper" in loc else "bottom"
    ax.text(x, y, text, transform=ax.transAxes, fontsize=8.5, va=va, ha=ha,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85,
                      edgecolor="#cccccc"))


def plot_distribution(dist: pd.Series, title: str, size: int,
                      bar_color: str = "steelblue", save_path: str = None):
    """
    Two-panel plot: frequency bars + residuals, with a five-number summary box.
    Works for both classical and quantum distributions.
    """
    expected  = size / len(dist)
    residuals = dist - expected
    summary   = five_number_summary(dist, size)
    std_expected = math.sqrt(expected * (1 - 1 / len(dist)))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)

    # ── Top: frequency distribution ───────────────────────────────────────────
    dist.plot(kind="bar", ax=ax1, color=bar_color, edgecolor="white", width=0.85)
    ax1.axhline(expected, color="tomato", linewidth=1.4, linestyle="--",
                label=f"Expected uniform ({expected:.0f}/bin)")
    ax1.set_ylabel("Frequency", fontsize=11)
    ax1.tick_params(axis="x", labelbottom=False)
    ax1.legend(fontsize=9)
    ax1.spines[["top", "right"]].set_visible(False)
    add_summary_box(ax1, summary, expected, loc="upper right")

    # Label each bar with its count, rotated vertically inside the bar
    for patch, val in zip(ax1.patches, dist.values):
        bar_height = patch.get_height()
        ax1.text(
            patch.get_x() + patch.get_width() / 2,  # horizontal center
            bar_height * 0.5,                        # halfway up the bar
            str(val),
            ha="center", va="center",
            fontsize=4.5, color="white", fontweight="bold",
            rotation=90,
        )

    # ── Bottom: residuals ─────────────────────────────────────────────────────
    res_colors = [bar_color if v >= 0 else "tomato" for v in residuals]
    ax2.bar(range(len(residuals)), residuals.values,
            color=res_colors, edgecolor="white", width=0.85)
    ax2.axhline(0, color="tomato", linewidth=1.4, linestyle="--")
    ax2.set_ylabel("Residual\n(Observed − Expected)", fontsize=10)
    ax2.set_xlabel("Number", fontsize=11)
    ax2.set_xticks(range(len(dist)))
    ax2.set_xticklabels(dist.index, fontsize=7, rotation=90)
    ax2.spines[["top", "right"]].set_visible(False)

    ax2.text(0.99, 0.95,
             f"Residual std={residuals.std():.1f}  (expected ~{std_expected:.1f})\n"
             f"max={residuals.max():+.0f}  min={residuals.min():+.0f}",
             transform=ax2.transAxes, fontsize=9, va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       alpha=0.85, edgecolor="#cccccc"))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    START, END, SIZE = 1, 50, 100_000

    # ── Classical distribution ────────────────────────────────────────────────
    print("Generating classical distribution...")
    classical_dist = generate_classical_distribution(START, END, SIZE)
    plot_distribution(classical_dist,
                      title="Distribution of Classical Random Numbers (Python random)",
                      size=SIZE,
                      bar_color="mediumseagreen",
                      save_path="classical_distribution.png")

    # ── Quantum distribution ──────────────────────────────────────────────────
    t0 = time.perf_counter()
    quantum_dist = generate_distribution(START, END, SIZE)
    elapsed = time.perf_counter() - t0
    print(f"Generated {SIZE} quantum numbers in {elapsed*1000:.0f} ms")

    plot_distribution(quantum_dist,
                      title="Distribution of Quantum Random Numbers (Qiskit Aer)",
                      size=SIZE,
                      bar_color="steelblue",
                      save_path="quantum_distribution.png")