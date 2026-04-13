"""Monteiro-Svaiter (MS) acceleration framework (Appendix B, Algorithm 3).

Implements the bisection-free acceleration scheme from Carmon et al. (2022),
adapted to custom Euclidean geometries as described in Appendix B.

Two modes of operation:

  p = inf (Theorem 1 / Algorithm 1):
    The ball optimization oracle solves  min f(x)  s.t.  ||x - q||_M <= r
    for a fixed radius r. The MS oracle is constructed by running the ball
    oracle and extracting the approximate stationarity condition. This gives
    an (inf, c)-movement bound, yielding (c||x0 - x*||_M)^{2/3} / eps^{2/3}
    outer iterations.

  2 <= p < inf (Theorem 2 / Algorithm 5):
    The proximal oracle solves  min f(x) + C_p ||x - q||_M^p  for a
    regularization strength tied to lambda. This gives a (p-1, c)-movement
    bound, yielding the rate from Theorem 2.

Both modes use Algorithm 3 (OptimalMSAcceleration) as the outer loop.
"""

from __future__ import annotations

import math

import numpy as np


def ms_accelerate(
    f_value,
    f_grad,
    ms_oracle,
    x0: np.ndarray,
    M: np.ndarray,
    T: int,
    s: float = np.inf,
    sigma: float = 0.5,
) -> list[np.ndarray]:
    """Run Algorithm 3 (OptimalMSAcceleration).

    This is the outer acceleration loop that takes an MS oracle and iterates
    it to get faster convergence than naive proximal iteration.

    Args:
        f_value: callable(x) -> float, evaluates the objective.
        f_grad: callable(x) -> ndarray, evaluates the gradient.
        ms_oracle: callable(q, lambda_prime) -> (x_tilde, lambda_val).
            Must satisfy the sigma-MS oracle condition (Definition B.1):
              ||x_tilde - q + (1/lambda) M^{-1} grad f(x_tilde)||_M
                  <= sigma * ||x_tilde - q||_M
        x0: Initial point.
        M: Positive definite geometry matrix (d x d). Defines ||x||_M = sqrt(x^T M x).
        T: Number of outer iterations.
        s: Movement bound order. s=inf for p=inf case, s=p-1 for finite p.
        sigma: MS oracle quality parameter (< 1).

    Returns:
        List of iterates [x0, x1, ..., xT].
    """
    d = len(x0)

    # Precompute M^{-1} for the v update.
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        M_inv = np.linalg.pinv(M)

    # alpha = exp(3 - 2/(s+1))  per Theorem B.3
    if s == np.inf:
        alpha_val = math.exp(3.0)
    else:
        alpha_val = math.exp(3.0 - 2.0 / (s + 1.0))

    # Initialize (Algorithm 3, lines 1-2).
    v = x0.copy()
    x = x0.copy()
    A_val = 0.0
    A_prime = 0.0

    # First oracle call to set lambda_1.
    x_tilde, lambda_val = ms_oracle(x0, 1.0)
    lambda_prime = lambda_val

    iterates = [x0.copy()]

    for t in range(T):
        # Line 4: a'_{t+1}
        disc = 1.0 + 4.0 * lambda_prime * A_val
        a_prime = (1.0 / (2.0 * lambda_prime)) * (1.0 + math.sqrt(max(disc, 0.0)))

        # Line 5: A'_{t+1}
        A_prime_new = A_val + a_prime

        # Line 6: query point q_t
        if A_prime_new > 0:
            qt = (A_val / A_prime_new) * x + (a_prime / A_prime_new) * v
        else:
            qt = x.copy()

        # Line 7: call MS oracle
        x_tilde, lambda_new = ms_oracle(qt, lambda_prime)

        # Line 8: adjustment (skip for t=0)
        if t > 0:
            gamma = min(1.0, lambda_prime / max(lambda_new, 1e-30))
        else:
            gamma = 1.0

        # Line 9: adjusted step sizes
        a_new = gamma * a_prime
        A_new = A_val + a_new

        # Line 10: update x
        if A_new > 0:
            x_new = ((1.0 - gamma * a_prime / A_prime_new) * A_val / A_new) * x \
                   + (gamma * a_prime / A_prime_new * A_val / A_new + a_new / A_new) * x_tilde \
                   if A_val > 0 else x_tilde.copy()
            # Simpler equivalent: convex combination
            if A_new > 1e-30:
                w_old = (A_new - a_new) / A_new  # = A_val / A_new
                # x_{t+1} = w_old * x_t + (1 - w_old) * [(gamma * A'_{t+1} / A_{t+1}) * x_tilde
                #            + ((1-gamma)*A_t / A_{t+1}) * x_t]
                # From line 10: x_{t+1} = ((1-gamma)*A_t / A_{t+1}) * x_t
                #                        + (gamma * A'_{t+1} / A_{t+1}) * x_tilde
                coeff_x = (A_new - gamma * A_prime_new) / max(A_new, 1e-30)
                coeff_xt = (gamma * A_prime_new) / max(A_new, 1e-30)
                x_new = coeff_x * x + coeff_xt * x_tilde
            else:
                x_new = x_tilde.copy()
        else:
            x_new = x_tilde.copy()

        # Lines 11-13: update lambda_prime
        if gamma >= 1.0 - 1e-12:
            lambda_prime_new = lambda_prime / alpha_val
        else:
            lambda_prime_new = lambda_prime * alpha_val

        # Line 14: update v
        grad_xt = f_grad(x_tilde)
        if grad_xt is not None and np.all(np.isfinite(grad_xt)):
            v = v - a_new * (M_inv @ grad_xt)
        # else keep v unchanged

        # Advance state.
        x = x_new
        A_val = A_new
        A_prime = A_prime_new
        lambda_prime = lambda_prime_new
        lambda_val = lambda_new

        iterates.append(x.copy())

    return iterates


# ---------------------------------------------------------------------------
# MS oracle construction for p = inf (Algorithm 1, lines 7-8)
# ---------------------------------------------------------------------------

def make_ms_oracle_robust(
    ball_oracle_fn,
    f_grad,
    M: np.ndarray,
    radius: float,
):
    """Construct a sigma-MS oracle from a ball optimization oracle (p = inf).

    Following (Carmon et al., 2020, Proposition 5):
    Given a (r, delta)-ball optimization oracle that solves
        min f(x)  s.t.  ||x - q||_M <= r
    we construct a sigma-MS oracle by:
        1. Call the ball oracle at query q with radius r
        2. Set lambda = 1/r (the movement bound is (inf, 1/r))
        3. Return (x_tilde, lambda)

    The MS condition ||x - q + (1/lambda) M^{-1} grad f(x)||_M <= sigma ||x - q||_M
    is approximately satisfied when the ball oracle returns an approximate minimizer.

    Args:
        ball_oracle_fn: callable(center, R) -> x_approx.
            Approximately solves min f(x) s.t. ||x - center||_M <= R.
        f_grad: callable(x) -> gradient of f at x.
        M: Geometry matrix.
        radius: Fixed ball radius r for all oracle calls.
    """
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        M_inv = np.linalg.pinv(M)

    def ms_oracle(q, lambda_prime):
        # Use the radius as the trust region size.
        # The lambda from the ball oracle is 1/r per the movement bound.
        x_tilde = ball_oracle_fn(q, radius)
        if x_tilde is None or not np.all(np.isfinite(x_tilde)):
            return q.copy(), lambda_prime

        lam = 1.0 / max(radius, 1e-16)
        return x_tilde, lam

    return ms_oracle


# ---------------------------------------------------------------------------
# MS oracle construction for 2 <= p < inf (Algorithm 5, line 3 / Algorithm 4)
# ---------------------------------------------------------------------------

def make_ms_oracle_gp(
    proximal_oracle_fn,
    f_grad,
    M: np.ndarray,
    p: float,
):
    """Construct a sigma-MS oracle from a proximal oracle (2 <= p < inf).

    Following Algorithm 4 / Lemma D.20:
    The proximal oracle solves
        min  f(x) + C_p ||x - q||_M^p
    where C_p = e * p^{p+1}.

    The MS oracle returns (x_tilde, lambda) where:
        lambda = C_p * ||x_tilde - q||_M^{p-2}
    which gives a (p-1, C_p^{1/(p-1)})-movement bound.

    Args:
        proximal_oracle_fn: callable(q, Cp) -> x_approx.
            Approximately solves min f(x) + Cp * ||x - q||_M^p.
        f_grad: callable(x) -> gradient of f at x.
        M: Geometry matrix.
        p: The norm parameter (>= 2).
    """
    Cp = math.e * p ** (p + 1)

    def ms_oracle(q, lambda_prime):
        x_tilde = proximal_oracle_fn(q, Cp)
        if x_tilde is None or not np.all(np.isfinite(x_tilde)):
            return q.copy(), lambda_prime

        diff = x_tilde - q
        norm_M = math.sqrt(max(float(diff @ M @ diff), 1e-30))
        lam = Cp * norm_M ** (p - 2)
        lam = max(lam, 1e-16)

        return x_tilde, lam

    return ms_oracle
