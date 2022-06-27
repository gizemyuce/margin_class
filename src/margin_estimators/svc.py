import cvxpy as cp
import numpy as np

from typing import Union

def solve_svc_problem(
    x: np.ndarray, y: np.ndarray, p: float = 1.0, solver: Union[None, str] = "MOSEK"
):
    """Solve the linear Support Vector Classifier problem
    argmin ||w||_p s.t. for all i: yi <xi, w> >= 1
    where ||.||_p is the L_p norm.
    """
    n, d = x.shape  # number of data points and data dimension

    w = cp.Variable(d)
    objective = cp.Minimize(cp.norm(w, p=p))
    constraints = [cp.multiply(y.squeeze(), x @ w) >= 1]
    prob = cp.Problem(objective, constraints)

    results = prob.solve(solver=solver, verbose=False)

    return results, prob, w