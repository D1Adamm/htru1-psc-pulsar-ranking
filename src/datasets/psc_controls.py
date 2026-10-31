"""Deterministic input transforms for the EXP-009 control and intervention checks.

Each transform runs on the native FvP/TvP arrays, before the shared resize and
normalization in ``arrays_to_tensor``.

The row shuffle is a phase-coherence control, not a complete morphology control.
It removes pulse alignment across rows but keeps blank rows, the pixel value
distribution, and the per-row share of bright bins, so width-like information
partly survives. A high score on these inputs is not itself a failure.
"""

from __future__ import annotations

import zlib

import numpy as np
import torch

from src.datasets.psc_preprocessing import arrays_to_tensor


def candidate_rng(plot_id: str, salt: str = "") -> np.random.Generator:
    """A generator fixed by candidate, so train and evaluation see identical inputs."""
    return np.random.default_rng(zlib.crc32(f"{plot_id}{salt}".encode()))


def shuffle_within_rows(array: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Independently permute the phase bins of each row."""
    out = array.copy()
    for i in range(out.shape[0]):
        out[i] = out[i][rng.permutation(out.shape[1])]
    return out


def shuffle_whole_array(array: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute every pixel, which also destroys blank rows."""
    return rng.permutation(array.ravel()).reshape(array.shape)


def roll_phase(array: np.ndarray, shift: int) -> np.ndarray:
    """Rotate pulse phase. A pulse has no preferred phase, so this should not matter."""
    return np.roll(array, shift, axis=1)


def rebin_phase(array: np.ndarray, factor: int = 4) -> np.ndarray:
    """Average adjacent phase bins, keeping the same signal at coarser resolution."""
    usable = array.shape[1] - array.shape[1] % factor
    return array[:, :usable].reshape(array.shape[0], usable // factor, factor).mean(2)


def fill_blank_rows(array: np.ndarray) -> np.ndarray:
    """Replace constant rows with the nearest varying row; unchanged if none vary."""
    blank = np.all(array == array[:, :1], axis=1)
    if blank.all() or not blank.any():
        return array.copy()
    good = np.where(~blank)[0]
    out = array.copy()
    for i in np.where(blank)[0]:
        out[i] = array[good[np.argmin(np.abs(good - i))]]
    return out


def row_shuffled_tensor(fvp: np.ndarray, tvp: np.ndarray, plot_id: str) -> torch.Tensor:
    """The C1 input: shuffle FvP then TvP within rows, then the normal pipeline."""
    rng = candidate_rng(plot_id)
    return arrays_to_tensor(shuffle_within_rows(fvp, rng), shuffle_within_rows(tvp, rng))


def whole_shuffled_tensor(fvp: np.ndarray, tvp: np.ndarray, plot_id: str) -> torch.Tensor:
    rng = candidate_rng(plot_id, "whole")
    return arrays_to_tensor(shuffle_whole_array(fvp, rng), shuffle_whole_array(tvp, rng))


def phase_rolled_tensor(fvp: np.ndarray, tvp: np.ndarray, plot_id: str) -> torch.Tensor:
    """Both channels share one fractional shift, as a real phase offset would."""
    fraction = candidate_rng(plot_id, "roll").random()
    return arrays_to_tensor(
        roll_phase(fvp, int(fraction * fvp.shape[1])),
        roll_phase(tvp, int(fraction * tvp.shape[1])),
    )


def rebinned_tensor(fvp: np.ndarray, tvp: np.ndarray, plot_id: str) -> torch.Tensor:
    """Coarsen only arrays with bins to spare; others are returned as usual."""
    if fvp.shape[1] < 100 or tvp.shape[1] < 100:
        return arrays_to_tensor(fvp, tvp)
    return arrays_to_tensor(rebin_phase(fvp), rebin_phase(tvp))


def blank_filled_tensor(fvp: np.ndarray, tvp: np.ndarray, plot_id: str) -> torch.Tensor:
    return arrays_to_tensor(fill_blank_rows(fvp), fill_blank_rows(tvp))


def fvp_removed_tensor(fvp: np.ndarray, tvp: np.ndarray, plot_id: str) -> torch.Tensor:
    image = arrays_to_tensor(fvp, tvp)
    image[0] = 0.0
    return image


def tvp_removed_tensor(fvp: np.ndarray, tvp: np.ndarray, plot_id: str) -> torch.Tensor:
    image = arrays_to_tensor(fvp, tvp)
    image[1] = 0.0
    return image


INTERVENTIONS = {
    "row_shuffled": row_shuffled_tensor,
    "whole_shuffled": whole_shuffled_tensor,
    "phase_rolled": phase_rolled_tensor,
    "rebinned": rebinned_tensor,
    "blank_filled": blank_filled_tensor,
    "fvp_removed": fvp_removed_tensor,
    "tvp_removed": tvp_removed_tensor,
}
