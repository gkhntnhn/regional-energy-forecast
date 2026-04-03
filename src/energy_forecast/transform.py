"""Box-Cox target transform utilities.

Provides forward and inverse Box-Cox transforms for stabilizing
variance in consumption data (heteroskedastic -> homoskedastic).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def boxcox_transform(y: NDArray[Any], lam: float) -> NDArray[Any]:
    """Box-Cox forward transform.

    Args:
        y: Positive-valued array (e.g. consumption in MWh).
        lam: Box-Cox lambda parameter.

    Returns:
        Transformed array in Box-Cox space.
    """
    if lam == 0:
        return np.log(y)  # type: ignore[no-any-return]
    result: NDArray[Any] = (np.power(y, lam) - 1) / lam
    return result


def inv_boxcox(y_bc: NDArray[Any], lam: float) -> NDArray[Any]:
    """Box-Cox inverse transform with numerical safety guard.

    Args:
        y_bc: Array in Box-Cox space.
        lam: Box-Cox lambda parameter (must match forward transform).

    Returns:
        Array in original space (MWh).
    """
    if lam == 0:
        return np.exp(y_bc)  # type: ignore[no-any-return]
    inner = lam * y_bc + 1
    # Guard: lam * y_bc + 1 < 0 -> NaN risk.
    # For lambda=0.39, y_bc < -2.56 triggers this — impossible in normal predictions.
    inner = np.maximum(inner, 1e-10)
    out: NDArray[Any] = np.power(inner, 1.0 / lam)
    return out
