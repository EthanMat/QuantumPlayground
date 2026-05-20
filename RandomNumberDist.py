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
        raise ValueError("Start must be less than or equal   to end.")
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
    from scipy import stats
    expected = float(size) / len(v)
    chi2, p = stats.chisquare(f_obs=v, f_exp=np.full_like(v, expected, dtype=float))
    return {
        "Min":      int(v.min()),
        "Q1":       float(np.percentile(v, 25)),
        "Median":   float(np.median(v)),
        "Q3":       float(np.percentile(v, 75)),
        "Max":      int(v.max()),
        "Mean":     float(v.mean()),
        "Std":      float(v.std()),
        "Chi2":     float(chi2),
        "p":        float(p),
        "dof":      int(len(v) - 1),
    }


def add_summary_box(ax, summary: dict, expected: float, loc: str = "upper right"):
    """Overlay a five-number summary text box on ax."""
    lines = [
        "Five-Number Summary (bin counts)",
        f"  Min    {summary['Min']}",
        f"  Q1     {summary['Q1']:.1f}",
        f"  Median {summary['Median']:.1f}",
        f"  Q3     {summary['Q3']:.1f}",
        f"  Max    {summary['Max']}",
        f"  Mean   {summary['Mean']:.1f}  (expected {expected:.0f})",
        f"  Std    {summary['Std']:.1f}",
        f"  Chi2   {summary['Chi2']:.2f}  (df={summary['dof']})",
        f"  p      {summary['p']:.3g}",
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
    Display a chi-squared table for a distribution and annotate the result.
    """
    expected     = size / len(dist)
    summary      = five_number_summary(dist, size)
    observed     = dist.values
    expected_arr = np.full_like(observed, expected, dtype=float)
    residuals    = observed - expected_arr
    chi2_terms   = (residuals ** 2) / expected_arr

    labels = [str(x) for x in dist.index]
    table_data = np.column_stack([
        observed,
        expected_arr,
        residuals,
        chi2_terms,
    ])
    col_labels = ["Observed", "Expected", "Residual", "Chi2 term"]

    n_rows = len(dist)
    fig_height = max(4, 0.28 * n_rows)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.96)
    ax.axis("off")

    add_summary_box(ax, summary, expected, loc="upper right")

    table = ax.table(
        cellText=np.round(table_data, 2).tolist(),
        rowLabels=labels,
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    START, END, SIZE = 1, 50, 10_000

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