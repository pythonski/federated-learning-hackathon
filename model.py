from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional, Tuple

import numpy as np


def _normal_cdf(x: np.ndarray | float) -> np.ndarray | float:
    """
    Fast approximation to the standard normal CDF Φ(x), vectorized.

    Uses the classic Abramowitz-Stegun style rational approximation:
      Φ(x) ≈ 1 - φ(x) * poly(t),  t = 1 / (1 + p x),  for x >= 0
    with symmetry Φ(-x) = 1 - Φ(x).

    This avoids np.vectorize(math.erf), which is extremely slow in sweeps.
    """
    x_arr = np.asarray(x, dtype=float)
    sign = np.sign(x_arr)
    ax = np.abs(x_arr)

    # Coefficients
    p = 0.2316419
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429

    t = 1.0 / (1.0 + p * ax)
    poly = (((((b5 * t + b4) * t) + b3) * t + b2) * t + b1) * t
    phi = (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * ax * ax)
    cdf_pos = 1.0 - phi * poly

    # reflect for negatives
    cdf = np.where(sign >= 0, cdf_pos, 1.0 - cdf_pos)
    if np.isscalar(x):
        return float(cdf)  # type: ignore[return-value]
    return cdf


def gini(values: np.ndarray) -> float:
    """
    Gini coefficient for nonnegative values.
    Uses the pairwise-difference definition for clarity (N is small here).
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError("gini() expects a 1D array")
    n = x.size
    if n == 0:
        return float("nan")
    mean = float(x.mean())
    if mean == 0.0:
        return 0.0
    diffs = np.abs(x[:, None] - x[None, :]).sum()
    return float(diffs / (2.0 * n * n * mean))


@dataclass(frozen=True)
class ModelParams:
    # time
    N: int
    T: int

    # compute / pledges
    hat_c: np.ndarray  # shape (N,)

    # capability scaling
    lam: float
    gamma: float
    sigma: float

    # verification
    v: float
    tau0: np.ndarray  # baseline std dev, shape (N,)
    beta: np.ndarray  # shape (N,)
    alpha: float
    z_alpha: float

    # payoffs
    theta: np.ndarray  # shape (N,)
    kappa: np.ndarray  # shape (N,)
    penalty: np.ndarray  # shape (N,)

    # behavioral assumption for best response
    others_expectation: Literal["pledge", "last_action"] = "pledge"
    a_grid_multiplier: float = 2.0
    a_grid_points: int = 301
    cap_at_pledge: bool = False  # if True, actors cannot exceed their pledge


def with_penalty_scale(params: ModelParams, scale: float) -> ModelParams:
    """Return a copy of params with penalties multiplied by scale."""
    return replace(params, penalty=params.penalty * float(scale))


def with_cap_at_pledge(params: ModelParams, cap: bool = True) -> ModelParams:
    """Return a copy of params with cap_at_pledge set."""
    return replace(params, cap_at_pledge=cap)


def load_params(path: str | Path) -> ModelParams:
    p = Path(path)
    data = json.loads(p.read_text())

    N = int(data["time"]["N"])
    T = int(data["time"]["T"])

    hat_c = np.asarray(data["compute"]["pledged_compute_hat_c_i_flop"], dtype=float)
    tau0 = np.asarray(data["verification"]["tau_i0"], dtype=float)
    beta = np.asarray(data["verification"]["beta_i"], dtype=float)
    theta = np.asarray(data["payoffs"]["theta_i"], dtype=float)
    kappa = np.asarray(data["payoffs"]["kappa_i"], dtype=float)
    penalty = np.asarray(data["payoffs"]["penalty_P_i"], dtype=float)

    def _check_len(name: str, arr: np.ndarray) -> None:
        if arr.shape != (N,):
            raise ValueError(f"{name} must have length N={N}, got shape {arr.shape}")

    _check_len("hat_c", hat_c)
    _check_len("tau0", tau0)
    _check_len("beta", beta)
    _check_len("theta", theta)
    _check_len("kappa", kappa)
    _check_len("penalty", penalty)

    lam = float(data["capability"]["scaling_law"]["lambda"])
    gamma = float(data["capability"]["scaling_law"]["gamma"])
    sigma = float(data["capability"]["observation"]["sigma"])

    v_block = data["verification"]["v_t"]
    v = float(v_block["value"])

    alpha = float(data["verification"]["test"]["alpha"])
    z_alpha = float(data["verification"]["test"]["z_alpha"])

    # Optional behavioral knobs (safe defaults if absent).
    others_expectation = data.get("behavior", {}).get("others_expectation", "pledge")
    if others_expectation not in ("pledge", "last_action"):
        raise ValueError("behavior.others_expectation must be 'pledge' or 'last_action'")

    a_grid_multiplier = float(data.get("behavior", {}).get("a_grid_multiplier", 2.0))
    a_grid_points = int(data.get("behavior", {}).get("a_grid_points", 301))
    cap_at_pledge = bool(data.get("behavior", {}).get("cap_at_pledge", False))

    return ModelParams(
        N=N,
        T=T,
        hat_c=hat_c,
        lam=lam,
        gamma=gamma,
        sigma=sigma,
        v=v,
        tau0=tau0,
        beta=beta,
        alpha=alpha,
        z_alpha=z_alpha,
        theta=theta,
        kappa=kappa,
        penalty=penalty,
        others_expectation=others_expectation,  # type: ignore[arg-type]
        a_grid_multiplier=a_grid_multiplier,
        a_grid_points=a_grid_points,
        cap_at_pledge=cap_at_pledge,
    )


def tau2(params: ModelParams, v: float) -> np.ndarray:
    """Per-actor reporting variance τ_i^2(v)."""
    return (params.tau0 ** 2) / (1.0 + params.beta * v)


def prob_flag(params: ModelParams, a: np.ndarray, v: float) -> np.ndarray:
    """
    Pr(Z_i > z_alpha | a_i, v) with:
      r_i ~ Normal(a_i, τ_i^2(v))
      Z_i = (hat_c_i - r_i) / sqrt(τ_i^2(v)) ~ Normal((hat_c_i - a_i)/τ_i(v), 1)
    """
    t2 = tau2(params, v)
    mu = (params.hat_c - a) / np.sqrt(t2)
    # If Z ~ N(mu, 1), then Pr(Z > z) = 1 - Φ(z - mu)
    return 1.0 - _normal_cdf(params.z_alpha - mu)


def benefit(params: ModelParams, q: float) -> np.ndarray:
    return params.theta * math.log1p(q)


def cost(params: ModelParams, a: np.ndarray) -> np.ndarray:
    return params.kappa * a


def _best_response_actions(
    params: ModelParams,
    I_prev: float,
    v: float,
    a_last: Optional[np.ndarray],
    cap_at_pledge: bool = False,
) -> np.ndarray:
    """
    Myopic per-round best response under a simple expectation about others.
    Each actor chooses a_i to maximize:
      E[u_i] = theta_i * log(1 + q(I_prev + A_other + a_i)) - kappa_i * a_i - P_i * Pr(flag)

    If cap_at_pledge=True, actions are constrained to a_i <= hat_c_i.
    """
    N = params.N

    if params.others_expectation == "last_action" and a_last is not None:
        a_expected = a_last.copy()
    else:
        a_expected = params.hat_c.copy()

    actions = np.zeros(N, dtype=float)
    a_max_global = float(max(params.hat_c.max(), 1.0) * params.a_grid_multiplier)
    # Precompute tau for this v once per call.
    tau = np.sqrt(tau2(params, v))

    for i in range(N):
        A_other = float(a_expected.sum() - a_expected[i])

        # Per-actor grid: cap at hat_c_i if requested
        if cap_at_pledge:
            a_max_i = float(params.hat_c[i])
        else:
            a_max_i = a_max_global
        grid = np.linspace(0.0, a_max_i, params.a_grid_points, dtype=float)

        # Evaluate expected utility on a grid and pick the best.
        q_grid = params.lam * np.power(I_prev + A_other + grid, params.gamma)
        b_grid = params.theta[i] * np.log1p(q_grid)
        c_grid = params.kappa[i] * grid

        # Flag probability for actor i at each candidate grid point.
        mu_i = (params.hat_c[i] - grid) / float(tau[i])
        p_flag_i = 1.0 - _normal_cdf(params.z_alpha - mu_i)
        penalty_grid = params.penalty[i] * p_flag_i

        eu = b_grid - c_grid - penalty_grid
        best_idx = int(np.argmax(eu))
        actions[i] = float(grid[best_idx])

    return actions


@dataclass
class SimulationResult:
    v: float
    actions: np.ndarray  # (T, N)
    flags: np.ndarray  # (T, N) bool
    utilities: np.ndarray  # (T, N)
    q: np.ndarray  # (T,)
    o: np.ndarray  # (T,)


def simulate(
    params: ModelParams,
    v: Optional[float] = None,
    seed: Optional[int] = None,
) -> SimulationResult:
    v_use = float(params.v if v is None else v)
    rng = np.random.default_rng(seed)

    T, N = params.T, params.N
    actions = np.zeros((T, N), dtype=float)
    flags = np.zeros((T, N), dtype=bool)
    utilities = np.zeros((T, N), dtype=float)
    q = np.zeros(T, dtype=float)
    o = np.zeros(T, dtype=float)

    I = 0.0
    a_last = None

    t2 = tau2(params, v_use)  # constant if v fixed
    tau = np.sqrt(t2)

    for t in range(T):
        a_t = _best_response_actions(params, I_prev=I, v=v_use, a_last=a_last, cap_at_pledge=params.cap_at_pledge)
        A_t = float(a_t.sum())
        I = I + A_t
        q_t = params.lam * (I**params.gamma)
        q[t] = q_t
        o[t] = q_t + float(rng.normal(0.0, params.sigma))

        # Verification reports and flags
        xi = rng.normal(0.0, tau, size=N)
        r = a_t + xi
        Z = (params.hat_c - r) / tau
        flagged = Z > params.z_alpha

        # Realized utility
        u = benefit(params, q_t) - cost(params, a_t) - params.penalty * flagged.astype(float)

        actions[t, :] = a_t
        flags[t, :] = flagged
        utilities[t, :] = u
        a_last = a_t

    return SimulationResult(v=v_use, actions=actions, flags=flags, utilities=utilities, q=q, o=o)


@dataclass
class ExpectedSimulationResult:
    v: float
    actions: np.ndarray  # (T, N)
    flag_prob: np.ndarray  # (T, N)
    expected_utilities: np.ndarray  # (T, N)
    q: np.ndarray  # (T,)


def simulate_expected(
    params: ModelParams,
    v: Optional[float] = None,
) -> ExpectedSimulationResult:
    """
    Deterministic expected-value simulation.

    - Actions are chosen via the same myopic expected-utility best response.
    - Flag rates use Pr(flag | a, v) (no Monte Carlo noise needed).
    - Utilities use expected penalty P_i * Pr(flag).
    """
    v_use = float(params.v if v is None else v)

    T, N = params.T, params.N
    actions = np.zeros((T, N), dtype=float)
    flag_prob = np.zeros((T, N), dtype=float)
    expected_utilities = np.zeros((T, N), dtype=float)
    q = np.zeros(T, dtype=float)

    I = 0.0
    a_last = None

    for t in range(T):
        a_t = _best_response_actions(params, I_prev=I, v=v_use, a_last=a_last, cap_at_pledge=params.cap_at_pledge)
        A_t = float(a_t.sum())
        I = I + A_t
        q_t = params.lam * (I**params.gamma)
        q[t] = q_t

        p_flag = prob_flag(params, a_t, v=v_use)
        u_exp = benefit(params, q_t) - cost(params, a_t) - params.penalty * p_flag

        actions[t, :] = a_t
        flag_prob[t, :] = p_flag
        expected_utilities[t, :] = u_exp
        a_last = a_t

    return ExpectedSimulationResult(v=v_use, actions=actions, flag_prob=flag_prob, expected_utilities=expected_utilities, q=q)


@dataclass(frozen=True)
class SweepMetrics:
    v_values: np.ndarray  # (V,)
    flag_rate: np.ndarray  # (V, N)
    expected_utility: np.ndarray  # (V, N)
    gini_compute: np.ndarray  # (V,)
    total_welfare: np.ndarray  # (V,)  (mean per-round sum of utilities)
    compliance_gap: np.ndarray  # (V, N) mean (a_i - hat_c_i) per actor
    cumulative_capability: np.ndarray  # (V,) final-period q(T)
    marginal_welfare: np.ndarray  # (V,) d(welfare)/dv (forward diff, last = 0)


def run_verification_sweep(
    params: ModelParams,
    v_values: Iterable[float],
    runs: int = 0,
    seed: int = 0,
) -> SweepMetrics:
    v_arr = np.asarray(list(v_values), dtype=float)
    V = v_arr.size
    N = params.N
    T = params.T

    flag_rate = np.zeros((V, N), dtype=float)
    expected_utility = np.zeros((V, N), dtype=float)
    gini_compute = np.zeros(V, dtype=float)
    total_welfare = np.zeros(V, dtype=float)
    compliance_gap = np.zeros((V, N), dtype=float)
    cumulative_capability = np.zeros(V, dtype=float)

    for vi, v in enumerate(v_arr):
        if runs <= 0:
            # Fast expected-value sweep (recommended default).
            res = simulate_expected(params, v=float(v))
            flag_rate[vi, :] = res.flag_prob.mean(axis=0)
            expected_utility[vi, :] = res.expected_utilities.mean(axis=0)
            gini_compute[vi] = gini(res.actions.sum(axis=0))
            total_welfare[vi] = float(res.expected_utilities.sum(axis=1).mean())
            # Compliance gap: mean (hat_c_i - a_i) over time (positive = under-contribution)
            compliance_gap[vi, :] = params.hat_c - res.actions.mean(axis=0)
            # Cumulative capability: final-period q(T)
            cumulative_capability[vi] = float(res.q[-1])
        else:
            # Monte Carlo sweep (slower; includes realized verification noise).
            base_rng = np.random.default_rng(seed + vi)
            seeds = base_rng.integers(0, 2**31 - 1, size=runs, dtype=np.int64)

            flags_sum = np.zeros(N, dtype=float)
            util_sum = np.zeros(N, dtype=float)
            welfare_sum = 0.0
            gini_sum = 0.0
            gap_sum = np.zeros(N, dtype=float)
            cap_sum = 0.0

            for r in range(runs):
                sim = simulate(params, v=float(v), seed=int(seeds[r]))
                flags_sum += sim.flags.mean(axis=0)
                util_sum += sim.utilities.mean(axis=0)
                gini_sum += gini(sim.actions.sum(axis=0))
                welfare_sum += float(sim.utilities.sum(axis=1).mean())
                gap_sum += params.hat_c - sim.actions.mean(axis=0)
                cap_sum += float(sim.q[-1])

            flag_rate[vi, :] = flags_sum / runs
            expected_utility[vi, :] = util_sum / runs
            gini_compute[vi] = gini_sum / runs
            total_welfare[vi] = welfare_sum / runs
            compliance_gap[vi, :] = gap_sum / runs
            cumulative_capability[vi] = cap_sum / runs

    # Marginal welfare: forward difference d(welfare)/dv
    marginal_welfare = np.zeros(V, dtype=float)
    if V > 1:
        dv = np.diff(v_arr)
        dw = np.diff(total_welfare)
        marginal_welfare[:-1] = dw / dv
        marginal_welfare[-1] = marginal_welfare[-2] if V > 1 else 0.0

    return SweepMetrics(
        v_values=v_arr,
        flag_rate=flag_rate,
        expected_utility=expected_utility,
        gini_compute=gini_compute,
        total_welfare=total_welfare,
        compliance_gap=compliance_gap,
        cumulative_capability=cumulative_capability,
        marginal_welfare=marginal_welfare,
    )

