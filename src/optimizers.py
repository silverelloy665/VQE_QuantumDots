"""
Optimizer wrappers and execution harness for VQE benchmark.
Includes: GD (Gradient Descent with optional momentum), ADAM, SPSA, and QNSPSA.
Uses vectorized batched PUB evaluations for exact central finite-difference gradients.
"""
import time
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit_algorithms.optimizers import SPSA, QNSPSA
from qiskit_algorithms.gradients import ReverseEstimatorGradient

def get_initial_point(init_name: str, num_params: int, seed: int = 42) -> np.ndarray:
    """
    Returns initial parameter vector for the specified initialization strategy.
    Options: 'zero', 'half', 'one', 'random'.
    'random' samples from U(0, 2*pi) — the full single-qubit rotation range.
    """
    init_lower = init_name.lower().strip()
    if init_lower == "zero":
        return np.zeros(num_params, dtype=float)
    elif init_lower == "half":
        return np.full(num_params, 0.5, dtype=float)
    elif init_lower == "one":
        return np.full(num_params, 1.0, dtype=float)
    elif init_lower == "random":
        rng = np.random.RandomState(seed)
        # Fixed: was U(0,1), now U(0, 2*pi) as documented in README
        return rng.uniform(0.0, 2.0 * np.pi, size=num_params).astype(float)
    else:
        raise ValueError(f"Unknown initialization strategy: {init_name}")

def compute_gradient_batched(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    params: np.ndarray,
    estimator: StatevectorEstimator | None = None,
    method: str = "finite_diff",
    eps: float = 1e-5,
    shift: float = np.pi / 2.0   # kept for backward-compat; ignored when method != 'param_shift_pi2'
) -> np.ndarray:
    """
    Vectorized batched gradient computation.

    Methods
    -------
    'finite_diff' (default):
        Vectorized central finite difference, evaluating all 2N parameter
        points in a single batched PUB call with step *eps*.
    'param_shift_pi4':
        Parameter-shift rule with shift=pi/4, valid for single-Pauli-frequency
        circuits (e.g. DexcG).  Formula:
            grad[i] = (E(+pi/4) - E(-pi/4)) / sqrt(2)
    'param_shift_pi2':
        Parameter-shift rule with shift=pi/2 (standard, kept for comparison).
        Formula: grad[i] = (E(+pi/2) - E(-pi/2)) / 2
    'reverse':
        Qiskit's ReverseEstimatorGradient; falls back to finite_diff if the
        circuit contains unsupported custom gates.
    """
    n = len(params)
    if n == 0:
        return np.array([], dtype=float)

    estimator = estimator or StatevectorEstimator()

    if method == "reverse":
        try:
            reg = ReverseEstimatorGradient()
            res = reg.run([circuit], [hamiltonian], [params]).result()
            return np.array(res.gradients[0], dtype=float)
        except Exception:
            # Fall back to exact central finite difference if reverse gradient cannot parse custom gates
            method = "finite_diff"

    if method in ("param_shift_pi4", "param_shift_pi2"):
        # Shift and denominator are fully determined by method name
        if method == "param_shift_pi4":
            delta = np.pi / 4.0
            denom = 1.0                   # For omega=2 (period pi): E(th+pi/4) - E(th-pi/4) = E'(th)
        else:  # param_shift_pi2
            delta = np.pi / 2.0
            denom = 2.0                   # 2 * sin(pi/2) = 2

        batch = []
        for i in range(n):
            p_plus = np.copy(params)
            p_minus = np.copy(params)
            p_plus[i] += delta
            p_minus[i] -= delta
            batch.append(p_plus)
            batch.append(p_minus)
        pub = (circuit, hamiltonian, batch)
        evs = estimator.run([pub]).result()[0].data.evs
        grad = np.zeros(n, dtype=float)
        for i in range(n):
            grad[i] = (evs[2 * i] - evs[2 * i + 1]) / denom
        return grad

    # Legacy alias kept for backward-compatibility
    if method == "param_shift":
        # Redirect to pi/4 shift (most physically correct for single-Pauli circuits)
        return compute_gradient_batched(circuit, hamiltonian, params, estimator,
                                        method="param_shift_pi4", eps=eps)

    # Default: Vectorized Central Finite Difference (Single Batched PUB)
    batch = []
    for i in range(n):
        p_plus = np.copy(params)
        p_minus = np.copy(params)
        p_plus[i] += eps
        p_minus[i] -= eps
        batch.append(p_plus)
        batch.append(p_minus)

    pub = (circuit, hamiltonian, batch)
    evs = estimator.run([pub]).result()[0].data.evs

    grad = np.zeros(n, dtype=float)
    denom = 2.0 * eps
    for i in range(n):
        grad[i] = (evs[2 * i] - evs[2 * i + 1]) / denom

    return grad

def run_gradient_descent(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    init_params: np.ndarray,
    estimator: StatevectorEstimator | None = None,
    maxiter: int = 50,
    lr: float = 0.05,
    momentum: float = 0.0,
    grad_method: str = "finite_diff",
    eps: float = 1e-5
) -> tuple[float, np.ndarray, list[float], float, dict]:
    """
    Gradient Descent with optional momentum:
    v_{t} = momentum * v_{t-1} + lr * ∇E(θ_{t-1})
    θ_{t} = θ_{t-1} - v_{t}
    """
    t0 = time.perf_counter()
    estimator = estimator or StatevectorEstimator()
    params = np.copy(init_params)
    v = np.zeros_like(params)
    n_params = len(params)

    pub_init = (circuit, hamiltonian, [params])
    current_e = float(estimator.run([pub_init]).result()[0].data.evs[0])
    history = [current_e]
    best_e = current_e
    best_it = 0
    best_params = np.copy(params)
    total_evals = 1

    for it in range(1, maxiter + 1):
        g = compute_gradient_batched(circuit, hamiltonian, params, estimator, method=grad_method, eps=eps)
        total_evals += 2 * n_params

        if momentum > 0.0:
            v = momentum * v + lr * g
            params = params - v
        else:
            params = params - lr * g

        current_e = float(estimator.run([(circuit, hamiltonian, [params])]).result()[0].data.evs[0])
        total_evals += 1
        history.append(current_e)

        if current_e < best_e:
            best_e = current_e
            best_it = it
            best_params = np.copy(params)

    wall_time = time.perf_counter() - t0
    final_e = current_e

    extra = {
        "best_energy": best_e,
        "best_iteration": best_it,
        "best_params": best_params,
        "total_evaluations": total_evals,
        "learning_rate": lr,
        "momentum": momentum
    }
    return final_e, params, history, wall_time, extra

def run_adam(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    init_params: np.ndarray,
    estimator: StatevectorEstimator | None = None,
    maxiter: int = 50,
    lr: float = 0.05,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    grad_method: str = "finite_diff",
    grad_eps: float = 1e-5
) -> tuple[float, np.ndarray, list[float], float, dict]:
    """
    ADAM Optimizer with vectorized batched finite-difference gradients.
    """
    t0 = time.perf_counter()
    estimator = estimator or StatevectorEstimator()
    params = np.copy(init_params)
    n_params = len(params)

    m = np.zeros_like(params)
    v = np.zeros_like(params)

    current_e = float(estimator.run([(circuit, hamiltonian, [params])]).result()[0].data.evs[0])
    history = [current_e]
    best_e = current_e
    best_it = 0
    best_params = np.copy(params)
    total_evals = 1

    for it in range(1, maxiter + 1):
        g = compute_gradient_batched(circuit, hamiltonian, params, estimator, method=grad_method, eps=grad_eps)
        total_evals += 2 * n_params

        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g ** 2)

        m_hat = m / (1.0 - beta1 ** it)
        v_hat = v / (1.0 - beta2 ** it)

        params = params - lr * m_hat / (np.sqrt(v_hat) + eps)
        current_e = float(estimator.run([(circuit, hamiltonian, [params])]).result()[0].data.evs[0])
        total_evals += 1
        history.append(current_e)

        if current_e < best_e:
            best_e = current_e
            best_it = it
            best_params = np.copy(params)

    wall_time = time.perf_counter() - t0
    final_e = current_e

    extra = {
        "best_energy": best_e,
        "best_iteration": best_it,
        "best_params": best_params,
        "total_evaluations": total_evals,
        "learning_rate": lr,
        "beta1": beta1,
        "beta2": beta2
    }
    return final_e, params, history, wall_time, extra

def run_spsa(
    energy_fn,
    init_params: np.ndarray,
    maxiter: int = 50,
    learning_rate: float = 0.1,
    perturbation: float = 0.1,
    seed: int = 42
) -> tuple[float, np.ndarray, list[float], float, dict]:
    """
    SPSA Optimizer with fixed learning rate and perturbation.
    """
    from qiskit_algorithms.utils import algorithm_globals
    algorithm_globals.random_seed = seed
    np.random.seed(seed)

    t0 = time.perf_counter()
    initial_e = float(energy_fn(init_params))
    history = [initial_e]
    eval_count = [1]

    def tracked_energy_fn(x):
        eval_count[0] += 1
        return energy_fn(x)

    def callback(nfev, x, fx, stepsize, accepted):
        history.append(float(fx))

    spsa = SPSA(
        maxiter=maxiter,
        learning_rate=learning_rate,
        perturbation=perturbation,
        callback=callback
    )

    res = spsa.minimize(tracked_energy_fn, init_params)
    final_e = float(tracked_energy_fn(res.x))
    wall_time = time.perf_counter() - t0

    while len(history) < maxiter + 1:
        history.append(final_e)
    if len(history) > maxiter + 1:
        history = history[:maxiter + 1]

    best_idx = int(np.argmin(history))
    best_e = float(history[best_idx])

    extra = {
        "best_energy": best_e,
        "best_iteration": best_idx,
        "best_params": res.x,
        "total_evaluations": eval_count[0],
        "learning_rate": learning_rate,
        "perturbation": perturbation
    }
    return final_e, res.x, history, wall_time, extra

def run_qnspsa(
    energy_fn,
    circuit: QuantumCircuit,
    init_params: np.ndarray,
    maxiter: int = 50,
    learning_rate: float = 0.1,
    perturbation: float = 0.1,
    sampler: StatevectorSampler | None = None,
    seed: int = 42
) -> tuple[float, np.ndarray, list[float], float, dict]:
    """
    QNSPSA with quantum state fidelity metric.
    """
    from qiskit_algorithms.utils import algorithm_globals
    algorithm_globals.random_seed = seed
    np.random.seed(seed)

    t0 = time.perf_counter()
    sampler = sampler or StatevectorSampler()
    initial_e = float(energy_fn(init_params))
    history = [initial_e]
    eval_count = [1]

    def tracked_energy_fn(x):
        eval_count[0] += 1
        return energy_fn(x)

    fidelity = QNSPSA.get_fidelity(circuit, sampler=sampler)

    def callback(nfev, x, fx, stepsize, accepted):
        history.append(float(fx))

    qnspsa = QNSPSA(
        fidelity=fidelity,
        maxiter=maxiter,
        learning_rate=learning_rate,
        perturbation=perturbation,
        callback=callback
    )

    res = qnspsa.minimize(tracked_energy_fn, init_params)
    final_e = float(tracked_energy_fn(res.x))
    wall_time = time.perf_counter() - t0

    while len(history) < maxiter + 1:
        history.append(final_e)
    if len(history) > maxiter + 1:
        history = history[:maxiter + 1]

    best_idx = int(np.argmin(history))
    best_e = float(history[best_idx])

    extra = {
        "best_energy": best_e,
        "best_iteration": best_idx,
        "best_params": res.x,
        "total_evaluations": eval_count[0],
        "learning_rate": learning_rate,
        "perturbation": perturbation
    }
    return final_e, res.x, history, wall_time, extra

def run_rotosolve_dexcg(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    estimator: StatevectorEstimator | None = None
) -> tuple[float, np.ndarray]:
    """
    Analytical single-parameter rotosolve for DexcG (1-parameter circuit).

    DexcG energy is a single-frequency sinusoidal:
        E(theta) = a + b*cos(2*theta) + c*sin(2*theta)

    Evaluate at theta = 0, pi/2, pi to extract a, b, c, then solve for
    optimal theta* analytically.

    Returns
    -------
    (E_opt, theta_opt) : (float, np.ndarray)
    """
    estimator = estimator or StatevectorEstimator()

    def eval_e(th):
        pub = (circuit, hamiltonian, [np.array([th])])
        return float(estimator.run([pub]).result()[0].data.evs[0])

    e0   = eval_e(0.0)
    epi2 = eval_e(np.pi / 2.0)
    epi  = eval_e(np.pi)

    # Solve for a, b, c:
    #   E(0)    = a + b
    #   E(pi/2) = a - b
    #   E(pi)   = a + b  (same as E(0) for 2-theta frequency)
    # Actually E(pi) = a + b*cos(2*pi) + c*sin(2*pi) = a + b
    # So b = (E(0) - E(pi/2)) / 2, a = (E(0) + E(pi/2)) / 2
    # For c we need a 4th point or use E(pi/4):
    epi4 = eval_e(np.pi / 4.0)
    # E(pi/4) = a + b*cos(pi/2) + c*sin(pi/2) = a + c
    a = (e0 + epi2) / 2.0
    b = (e0 - epi2) / 2.0
    c = epi4 - a

    # Optimal: theta* = 0.5 * arctan2(c, b) + pi/2 if b*cos + c*sin < 0 (minimum)
    # Minimum of a + b*cos(2θ) + c*sin(2θ) is at: 2θ* = arctan2(c, b) + pi
    # → θ* = 0.5 * (arctan2(c, b) + pi)
    theta_star = 0.5 * (np.arctan2(c, b) + np.pi)
    e_star = eval_e(theta_star)

    # Also check theta_star + pi/2 (the other candidate)
    theta_alt = theta_star + np.pi / 2.0
    e_alt = eval_e(theta_alt)
    if e_alt < e_star:
        theta_star = theta_alt
        e_star = e_alt

    return e_star, np.array([theta_star])

def run_vqe_single(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    ansatz_name: str,
    init_name: str,
    optimizer_name: str,
    maxiter: int = 50,
    lr: float | None = None,
    momentum: float = 0.0,
    estimator: StatevectorEstimator | None = None,
    sampler: StatevectorSampler | None = None,
    seed: int = 42
) -> dict:
    """
    Runs a single VQE configuration and returns a comprehensive dictionary of metrics.
    """
    estimator = estimator or StatevectorEstimator()
    sampler = sampler or StatevectorSampler()

    def cost_fn(theta):
        pub = (circuit, hamiltonian, [theta])
        job = estimator.run([pub])
        val = float(job.result()[0].data.evs[0])
        return val

    num_params = circuit.num_parameters
    init_params = get_initial_point(init_name, num_params, seed=seed)

    opt_upper = optimizer_name.upper().strip()
    if opt_upper == "GD":
        gd_lr = lr if lr is not None else 0.05
        final_e, final_p, hist, wtime, extra = run_gradient_descent(
            circuit, hamiltonian, init_params, estimator, maxiter=maxiter, lr=gd_lr, momentum=momentum
        )
    elif opt_upper == "ADAM":
        adam_lr = lr if lr is not None else 0.05
        final_e, final_p, hist, wtime, extra = run_adam(
            circuit, hamiltonian, init_params, estimator, maxiter=maxiter, lr=adam_lr
        )
    elif opt_upper == "SPSA":
        spsa_lr = lr if lr is not None else 0.1
        final_e, final_p, hist, wtime, extra = run_spsa(
            cost_fn, init_params, maxiter=maxiter, learning_rate=spsa_lr, perturbation=0.1, seed=seed
        )
    elif opt_upper == "QNSPSA":
        qnspsa_lr = lr if lr is not None else 0.1
        final_e, final_p, hist, wtime, extra = run_qnspsa(
            cost_fn, circuit, init_params, maxiter=maxiter, learning_rate=qnspsa_lr, perturbation=0.1, sampler=sampler, seed=seed
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return {
        "ansatz": ansatz_name,
        "initialization": init_name,
        "optimizer": opt_upper,
        "final_energy": final_e,
        "best_energy": extra["best_energy"],
        "best_iteration": extra["best_iteration"],
        "final_params": final_p.tolist() if isinstance(final_p, np.ndarray) else list(final_p),
        "best_params": extra["best_params"].tolist() if isinstance(extra["best_params"], np.ndarray) else list(extra["best_params"]),
        "energy_history": hist,
        "wall_time": wtime,
        "num_params": num_params,
        "total_evaluations": extra["total_evaluations"],
        "learning_rate": extra.get("learning_rate", lr)
    }
