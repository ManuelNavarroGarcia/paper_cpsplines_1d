import itertools
from typing import Dict, Iterable, Optional, Tuple, Union

import mosek.fusion
import numpy as np
from cpsplines.mosek_functions.interval_constraints import IntConstraints
from cpsplines.psplines.bspline_basis import BsplineBasis
from cpsplines.psplines.penalty_matrix import PenaltyMatrix
from cpsplines.utils.cholesky_semidefinite import cholesky_semidef
from cpsplines.utils.weighted_b import get_weighted_B
from scipy.linalg import block_diag


class NumericalError(Exception):
    pass


def gcv_multiple(
    obj_matrices: Dict[str, np.ndarray],
    B_weighted: np.ndarray,
    sp: np.ndarray,
) -> float:
    y_hat = np.zeros(obj_matrices["y"].shape)
    penalty_term = np.zeros(obj_matrices["D_mul"].shape)
    for i, s in enumerate(sp):
        y_hat[i, :] = (
            B_weighted
            @ np.linalg.solve(
                obj_matrices["B_mul"] + np.multiply(s, obj_matrices["D_mul"]),
                B_weighted.T,
            )
            @ obj_matrices["y"][i, :]
        )
        penalty_term += np.multiply(s, obj_matrices["D_mul"])
    return (
        np.linalg.norm((obj_matrices["y"] - y_hat)) ** 2
        * np.prod(obj_matrices["y"].shape)
    ) / (
        np.prod(obj_matrices["y"].shape)
        - np.trace(
            np.linalg.solve(
                np.multiply(obj_matrices["y"].shape[0], obj_matrices["B_mul"])
                + penalty_term,
                obj_matrices["B_mul"],
            )
        )
    ) ** 2


class MultipleCurves:
    def __init__(
        self,
        deg: int = 3,
        ord_d: int = 2,
        n_int: int = 10,
        x_range: Optional[Tuple[Union[int, float]]] = None,
        sp_grid: Optional[Iterable[Union[int, float]]] = None,
        overlap_thr: Optional[Iterable[Union[int, float]]] = None,
    ):
        self.deg = deg
        self.ord_d = ord_d
        self.n_int = n_int
        self.x_range = x_range
        self.sp_grid = sp_grid
        self.overlap_thr = overlap_thr

    def _get_bspline_basis(self, x: np.ndarray) -> BsplineBasis:
        x_min, x_max = np.min(x), np.max(x)
        prediction_dict = {}
        if self.x_range is not None:
            pred_min, pred_max = min(self.x_range), max(self.x_range)
            if pred_max > x_max:
                prediction_dict["forward"] = pred_max
            if pred_min < x_min:
                prediction_dict["backwards"] = pred_min
        bsp = BsplineBasis(
            deg=self.deg,
            xsample=np.sort(x),
            n_int=self.n_int,
            prediction=prediction_dict,
        )
        bsp.get_matrix_B()
        return bsp

    def _get_obj_func_arrays(
        self, x: np.ndarray, y: np.ndarray
    ) -> Dict[str, np.ndarray]:

        obj_matrices = {}
        obj_matrices["B"] = self.bspline_basis.matrixB
        obj_matrices["B_mul"] = obj_matrices["B"].T @ obj_matrices["B"]
        obj_matrices["D_mul"] = PenaltyMatrix(
            bspline=self.bspline_basis
        ).get_penalty_matrix(**{"ord_d": self.ord_d})
        y_ordered = y[:, np.argsort(x)]
        y_ext = np.zeros((y.shape[0], self.bspline_basis.matrixB.shape[0]))
        y_ext[
            :, self.bspline_basis.int_back : len(x) + self.bspline_basis.int_back
        ] = y_ordered
        obj_matrices["y"] = y_ext
        return obj_matrices

    def _initialize_model(
        self, n_groups: int, L_B: np.ndarray, L_D: np.ndarray, lin_term: np.ndarray
    ) -> mosek.fusion.Model:
        M = mosek.fusion.Model()
        theta = M.variable(
            "theta",
            self.bspline_basis.matrixB.shape[1] * n_groups,
            mosek.fusion.Domain.unbounded(),
        )
        t_B = M.variable("t_B", 1, mosek.fusion.Domain.greaterThan(0.0))
        t_D = M.variable("t_D", n_groups, mosek.fusion.Domain.greaterThan(0.0))
        sp_params = [M.parameter(f"sp_{i}", 1) for i in range(n_groups)]
        flatten_theta = mosek.fusion.Var.flatten(theta)
        M.constraint(
            "rot_cone_B",
            mosek.fusion.Expr.vstack(
                t_B,
                1 / 2,
                mosek.fusion.Expr.mul(mosek.fusion.Matrix.sparse(L_B.T), flatten_theta),
            ),
            mosek.fusion.Domain.inRotatedQCone(),
        )
        for g in range(n_groups):
            M.constraint(
                f"rot_cone_D_{g}",
                mosek.fusion.Expr.vstack(
                    t_D.slice(g, g + 1),
                    1 / 2,
                    mosek.fusion.Expr.mul(
                        mosek.fusion.Matrix.sparse(L_D.T),
                        flatten_theta.slice(g * L_D.shape[1], (g + 1) * L_D.shape[1]),
                    ),
                ),
                mosek.fusion.Domain.inRotatedQCone(),
            )

        obj = [mosek.fusion.Expr.dot(lin_term, flatten_theta)]
        for g, sp in enumerate(sp_params):
            obj.append(mosek.fusion.Expr.dot(sp, t_D.slice(g, g + 1)))
        obj = mosek.fusion.Expr.add(t_B, mosek.fusion.Expr.add(obj))
        obj = M.objective(
            "obj",
            mosek.fusion.ObjectiveSense.Minimize,
            obj,
        )
        return M

    def _get_sp_grid_search(
        self,
        obj_matrices: Dict[str, np.ndarray],
        B_weighted: np.ndarray,
    ) -> Tuple[Union[int, float]]:

        if self.sp_grid is None:
            self.sp_grid = [
                (0.01, 0.1, 1, 10) for _ in range(obj_matrices["y"].shape[0])
            ]
        iter_sp = list(itertools.product(*self.sp_grid))
        gcv = [
            gcv_multiple(
                obj_matrices=obj_matrices,
                B_weighted=B_weighted,
                sp=sp,
            )
            for sp in iter_sp
        ]
        return iter_sp[gcv.index(min(gcv))]

    def _overlap_cons(self, model: mosek.fusion.Model, n_groups: int):
        S = self.bspline_basis.get_matrices_S()
        n_intervals = self.bspline_basis.matrixB.shape[1] - self.deg
        if self.overlap_thr is None:
            self.overlap_thr = [0] * (n_groups - 1)
        int_cons = IntConstraints(
            bspline=[self.bspline_basis],
            var_name=0,
            derivative=0,
            constraints={"+": self.overlap_thr},
        )
        W = int_cons._get_matrices_W()
        C = [w @ s for w, s in zip(W, S)]
        H = int_cons._get_matrices_H()
        X = model.variable(
            mosek.fusion.Domain.inPSDCone(
                self.deg + 1,
                n_intervals * (n_groups - 1),
            )
        )
        ind_term = W[0][:, 0]
        theta = model.getVariable("theta")
        n_var = self.bspline_basis.matrixB.shape[1]
        for i, g in enumerate(range(n_groups - 1)):
            theta_1 = theta.slice(g * n_var, (g + 1) * n_var)
            theta_2 = theta.slice((g + 1) * n_var, (g + 2) * n_var)
            threshold = int_cons.constraints["+"][i]
            for j, w in enumerate(range(n_intervals)):
                actual_index = w + g * n_intervals
                theta_slice_1 = mosek.fusion.Expr.mul(
                    C[j], theta_1.slice(w, w + self.deg + 1)
                )
                theta_slice_2 = mosek.fusion.Expr.mul(
                    C[j], theta_2.slice(w, w + self.deg + 1)
                )
                slice_X = X.slice(
                    [actual_index, 0, 0],
                    [
                        actual_index + 1,
                        self.deg + 1,
                        self.deg + 1,
                    ],
                ).reshape([self.deg + 1, self.deg + 1])

                for k in range(self.deg):
                    model.constraint(
                        mosek.fusion.Expr.dot(H[0][k], slice_X),
                        mosek.fusion.Domain.equalsTo(0.0),
                    )

                for k in range(self.deg + 1):
                    model.constraint(
                        mosek.fusion.Expr.sub(
                            mosek.fusion.Expr.sub(
                                theta_slice_2.slice(k, k + 1),
                                theta_slice_1.slice(k, k + 1),
                            ),
                            mosek.fusion.Expr.dot(H[1][k], slice_X),
                        ),
                        mosek.fusion.Domain.equalsTo(ind_term[k] * threshold),
                    )
        return model

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.bspline_basis = self._get_bspline_basis(x=x)
        obj_matrices = self._get_obj_func_arrays(x=x, y=y)
        B_weighted = get_weighted_B(bspline_bases=(self.bspline_basis,))[0]
        obj_matrices["B_mul"] = B_weighted.T @ B_weighted

        lin_term = np.multiply(-2, obj_matrices["y"] @ B_weighted).flatten()
        L_B = cholesky_semidef(obj_matrices["B_mul"])
        L_B = block_diag(*[L_B] * obj_matrices["y"].shape[0])
        L_D = cholesky_semidef(obj_matrices["D_mul"])
        M = self._initialize_model(
            n_groups=obj_matrices["y"].shape[0], L_B=L_B, L_D=L_D, lin_term=lin_term
        )
        if self.overlap_thr is not None:
            M = self._overlap_cons(model=M, n_groups=obj_matrices["y"].shape[0])
        theta_shape = (obj_matrices["y"].shape[0], self.bspline_basis.matrixB.shape[1])
        self.best_sp = self._get_sp_grid_search(
            obj_matrices=obj_matrices, B_weighted=B_weighted
        )
        for i, sp in enumerate(self.best_sp):
            M.getParameter(f"sp_{i}").setValue(sp)
        try:
            M.solve()
            self.sol = M.getVariable("theta").level().reshape(theta_shape)
            self.y_fitted = (
                block_diag(*[obj_matrices["B"]] * obj_matrices["y"].shape[0])
                @ self.sol.flatten()
            ).reshape(obj_matrices["y"].shape)
        except mosek.fusion.SolutionError as e:
            raise NumericalError(f"The original error was {e}")
