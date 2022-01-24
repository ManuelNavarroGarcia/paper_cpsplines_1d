from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cpsplines.fittings.grid_cpsplines import GridCPsplines, NumericalError
from cpsplines.psplines.bspline_basis import BsplineBasis
from scipy.special import erf


def zoom_covid(
    x: np.ndarray,
    y: np.ndarray,
    y_curve: np.ndarray,
    zoom_pts: Tuple[int, int],
    alpha: Union[int, float] = 1,
    y_hline: Optional[Union[int, float]] = None,
    figsize: Tuple[Union[int, float]] = (15, 10),
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[matplotlib.figure.Figure, plt.axes]:

    fig, ax = plt.subplots(figsize=figsize)
    _ = ax.scatter(x=x[slice(*zoom_pts)], y=y[slice(*zoom_pts)], c="b", alpha=alpha)
    _ = ax.plot(x[slice(*zoom_pts)], y_curve[slice(*zoom_pts)], c="k", linewidth=2.0)
    if y_hline is not None:
        _ = ax.axhline(y_hline, color="red", linewidth=2.0, linestyle="--")
    if save_path is not None:
        _ = fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


def plot_bspline_basis(
    bsp: BsplineBasis,
    plot_knots: bool = False,
    bsp_depict: Optional[Iterable[int]] = None,
    figsize: Tuple[Union[int, float], Union[int, float]] = (15, 10),
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[matplotlib.figure.Figure, plt.axes]:

    # State the prediction step used to plot B-spline outside the covariate
    # domain and plot features of these B-spline basis elements
    PRED_STEP = np.ptp(bsp.xsample) / 1000
    LINE_DETAILS = {
        "continue": {"col": "grey", "linestyle": "solid"},
        "new": {"col": "k", "linestyle": "dashed"},
    }

    fig, ax = plt.subplots(figsize=figsize)
    # Create three basis matrix: one for the backwards prediction range, one for
    # the forward prediction range and the last, extracted from the original
    # basis matrix, for the covariate domain
    B = bsp.matrixB[bsp.int_back : bsp.int_back + len(bsp.xsample), :]
    if "forward" in bsp.prediction:
        _ = ax.axvline(x=bsp.xsample.max(), color="r", linewidth=2.0)
        x_forw = np.arange(bsp.prediction["forward"], bsp.xsample.max(), -PRED_STEP)[
            ::-1
        ]
        B_forw = bsp.bspline_basis(x=x_forw)
    if "backwards" in bsp.prediction:
        _ = ax.axvline(x=bsp.xsample.min(), color="r", linewidth=2.0)
        x_back = np.arange(bsp.prediction["backwards"], bsp.xsample.min(), PRED_STEP)
        B_back = bsp.bspline_basis(x=x_back)
    for i in range(bsp.n_int + bsp.int_forw + bsp.int_back + bsp.deg):
        # Plot in blue the elements in the covariate domain
        _ = ax.plot(bsp.xsample, B[:, i], c="b", linewidth=2.0)
        # Plot the elements on the forward prediction range
        if "forward" in bsp.prediction:
            if i < bsp.deg + bsp.n_int + bsp.int_back:
                details = LINE_DETAILS["continue"]
            else:
                details = LINE_DETAILS["new"]
            _ = ax.plot(
                x_forw,
                B_forw[:, i],
                c=details["col"],
                linewidth=2.0,
                linestyle=details["linestyle"],
            )
        # Plot the elements on the backwards prediction range
        if "backwards" in bsp.prediction:
            if i >= bsp.int_back:
                details = LINE_DETAILS["continue"]
            else:
                details = LINE_DETAILS["new"]
            _ = ax.plot(
                x_back,
                B_back[:, i],
                c=details["col"],
                linewidth=2.0,
                linestyle=details["linestyle"],
            )

    if plot_knots:
        for i in range(1, bsp.n_int + bsp.int_back + bsp.int_forw + 1):
            _ = ax.axvline(
                bsp.knots[bsp.deg + i], color="grey", linewidth=1.5, linestyle="--"
            )
    if bsp_depict is not None:
        for elem in bsp_depict:
            _ = ax.plot(bsp.xsample, B[:, elem], c="r", linewidth=2.0)
    # Remove the top and right axis
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if save_path is not None:
        _ = fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


def get_simulated_errors(
    func: Callable,
    n_iter: int = 100,
    sigma: Union[int, float] = 0.3,
    size: int = 100,
    first_seed: int = 0,
    constraints: Optional[Dict[int, Dict[int, Dict[str, Union[int, float]]]]] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    sin_ls = []
    con_ls = []
    if constraints is None:
        constraints = {}
    for i in range(first_seed, n_iter + first_seed):
        np.random.seed(i)
        # Generate the covariate samples from a uniform distribution
        x = np.sort(np.random.uniform(low=0.0, high=1.0, size=size))
        # Generate the errors from a normal distribution with standard deviation
        # `sigma`
        error = np.random.normal(loc=0.0, scale=sigma, size=size)
        y_teo = func(x)
        y = y_teo + error
        try:
            # Get the unconstrained fitted curve (and errors)
            sin = GridCPsplines(
                int_constraints={},
                **kwargs,
            )
            _ = sin.fit(x=(x,), y=y)
            sin_ls.append(sin.y_fitted - y_teo)

            # Get the constrained fitted curve (and errors)
            con = GridCPsplines(
                int_constraints=constraints,
                **kwargs,
            )
            _ = con.fit(x=(x,), y=y)
            con_ls.append(con.y_fitted - y_teo)
        except NumericalError:
            pass
    return pd.DataFrame(sin_ls), pd.DataFrame(con_ls)


def simulated_incr_function(x: np.ndarray) -> np.ndarray:

    return 5 + erf(15 * x - 3) + erf(30 * x - 12) + erf(45 * x - 27) + erf(60 * x - 48)


def simulated_multiple_function(x: np.ndarray) -> np.ndarray:

    return 1 / (1 + np.exp(-10 * x))


def simulated_nonneg_function(x: np.ndarray) -> np.ndarray:

    y_teo = np.exp(4 - x / 25) + 4 * np.cos(x / 8)
    y_teo += np.abs(np.min(y_teo))
    return y_teo


def report_Lp_distances(
    df: pd.DataFrame,
    ps: Iterable[Union[str, int, float]],
    factors: Iterable[Union[int, float]],
    ponder: bool = True,
    round_n: int = 3,
) -> pd.Series:

    ls = []
    for f, p in zip(factors, ps):
        dist = df.apply(np.linalg.norm, args=(p,), axis=1)
        if ponder and p is not np.inf:
            dist /= df.shape[1]
        ls.append(f * dist.mean())
        ls.append(f * dist.std())
    return pd.Series(ls).round(round_n)


def plot_confidence_bands(
    df: pd.DataFrame,
    x: Union[np.ndarray, pd.Series],
    y: Union[np.ndarray, pd.Series],
    y_teo: Union[np.ndarray, pd.Series],
    y_fitted: Union[np.ndarray, pd.Series],
    alpha_cb: Union[int, float] = 0.05,
    alpha_pts: Union[int, float] = 0.15,
    color_cb: str = "b",
    constrained: bool = True,
    figsize: Tuple[Union[int, float]] = (15, 10),
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[matplotlib.figure.Figure, plt.axes]:

    fig, ax = plt.subplots(figsize=figsize)
    lab = "Non-negative " if constrained else "Unconstrained "
    # Plot the theoretical curve and the plots used to fit the curve
    _ = ax.scatter(x, y, c="k", alpha=alpha_pts)
    _ = ax.plot(x, y_teo, c="k", label="Theoretical curve")
    # Plot the confidence bands
    _ = ax.fill_between(
        x,
        df.quantile(alpha_cb / 2),
        df.quantile(1 - alpha_cb / 2),
        alpha=0.15,
        color=color_cb,
        label=lab + "confidence bands",
    )
    # Plot a red line on the horizontal axis
    _ = ax.axhline(0, c="r")
    _ = ax.plot(x, y_fitted, c=color_cb, label=lab + "curve")
    _ = ax.tick_params(axis="both", which="major", labelsize=16)
    _ = ax.legend(prop={"size": 20})
    if save_path:
        _ = fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


def displaced_forecast_covid(
    deriv: np.ndarray,
    xmax: Union[int, float],
    x_pred: Union[int, float],
    lag: Union[int, float] = 0,
    factor_deriv: Union[int, float] = 1,
) -> Dict[str, Union[np.ndarray, Union[int, float]]]:

    x_pred_ext = np.arange(xmax + lag, x_pred + lag, lag)
    xmax_pred = x_pred_ext[np.where(x_pred_ext <= x_pred)]
    deriv_pred = np.array([elem * factor_deriv for elem in deriv])[: len(xmax_pred)]
    return {"x_pred": xmax_pred, "deriv_pred": deriv_pred}


def predict_covid(
    x: np.ndarray,
    y: np.ndarray,
    factors_dict: Dict[str, np.ndarray],
    x_pred: Union[int, float],
    deriv_range: slice,
    tol: Union[int, float],
    **kwargs,
) -> Dict[str, np.ndarray]:
    d = {}
    y_norm = (y - np.min(y)) / np.ptp(y)
    x_fore = np.arange(x.max() + 1, x_pred + 1, 1)
    no_pt_cons = GridCPsplines(**kwargs)
    no_pt_cons.fit(x=(x,), y=y_norm)
    y_ext = np.concatenate(
        (no_pt_cons.y_fitted[: len(y)], no_pt_cons.predict(x=[x_fore]))
    )
    derivatives = np.matmul(
        no_pt_cons.bspline_bases[0].bspline_basis.derivative(nu=1)(
            no_pt_cons.bspline_bases[0].xsample
        ),
        no_pt_cons.sol,
    )

    d["lag: 0, deriv: 0"] = np.ptp(y) * y_ext + np.min(y)
    for lag, deriv in zip(factors_dict["lag"], factors_dict["factor_deriv"]):
        pt_deriv = displaced_forecast_covid(
            xmax=x.max(),
            x_pred=x_pred,
            lag=lag,
            factor_deriv=deriv,
            deriv=derivatives[deriv_range],
        )
        y_fit = False
        while y_fit is False:
            try:
                pt_cons = GridCPsplines(
                    **kwargs,
                    pt_constraints={
                        (1,): ((pt_deriv["x_pred"],), pt_deriv["deriv_pred"], tol)
                    },
                )
                pt_cons.fit(x=(x,), y=y_norm)
            except:
                pass
            y_fit = hasattr(pt_cons, "y_fitted")
            tol += 0.001
        y_ext = np.concatenate(
            (pt_cons.y_fitted[: len(y)], pt_cons.predict(x=[x_fore]))
        )
        d[f"lag: {lag}, deriv: {deriv}"] = np.ptp(y) * y_ext + np.min(y)
    return d
