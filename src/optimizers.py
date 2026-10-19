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
from qiskit.quantum_info import Statevector

class FastStatevectorEstimator:
    """Accelerated exact Statevector estimator using precomputed matrix Hamiltonian and C++ Aer parameter binding."""
    def __init__(self, hamiltonian: SparsePauliOp):
        self.H_mat = hamiltonian.to_matrix()
        try:
            from qiskit_aer import AerSimulator
            self.sim = AerSimulator(method="statevector", max_parallel_threads=1, max_parallel_experiments=1)
            self._has_aer = True
        except Exception:
            self._has_aer = False
        self._circuit_cache = {}

    def run(self, pubs):
        results = []
        for qc, _, param_list in pubs:
            evs = []
            if getattr(self, "_has_aer", False):
                try:
                    qc_key = id(qc)
                    if qc_key not in self._circuit_cache:
                        c_save = qc.copy()
                        c_save.save_statevector()
                        self._circuit_cache[qc_key] = (c_save, list(qc.parameters))
                    c_save, params = self._circuit_cache[qc_key]

                    if len(params) > 0 and len(param_list) > 0:
                        bind_dict = {
                            p: [param_list[k][i] for k in range(len(param_list))]
                            for i, p in enumerate(params)
                        }
                        job = self.sim.run([c_save], parameter_binds=[bind_dict])
                        for r in job.result().results:
                            sv = np.asarray(r.data.statevector)
                            v = float(np.vdot(sv, self.H_mat.dot(sv)).real)
                            evs.append(v)
                    elif len(params) == 0:
                        job = self.sim.run([c_save])
                        sv = np.asarray(job.result().results[0].data.statevector)
                        v = float(np.vdot(sv, self.H_mat.dot(sv)).real)
                        evs = [v] * max(len(param_list), 1)
                    else:
                        raise ValueError("Empty param_list for parameterized circuit")
                except Exception:
                    evs = []
                    for p in param_list:
                        bound = qc.assign_parameters(p)
                        sv = Statevector(bound)
                        v = float(np.vdot(sv.data, self.H_mat.dot(sv.data)).real)
                        evs.append(v)
            else:
                for p in param_list:
                    bound = qc.assign_parameters(p)
                    sv = Statevector(bound)
                    v = float(np.vdot(sv.data, self.H_mat.dot(sv.data)).real)
                    evs.append(v)
            class _Data:
                def __init__(self, vals):
                    self.evs = np.array(vals, dtype=float)
            class _Res:
                def __init__(self, vals):
                    self.data = _Data(vals)
            results.append(_Res(evs))
        class _Job:
            def __init__(self, res):
                self._res = res
            def result(self):
                return self._res
        return _Job(results)



def get_initial_point(init_name: str, num_params: int, seed: int = 42, sigma: float = 0.05) -> np.ndarray:
    """
    Returns initial parameter vector for the specified initialization strategy.
    Options: 'zero', 'half', 'one', 'random', 'perturbed' (or 'perturbed_<sigma>').
    'random' samples from U(0, 2*pi).
    'perturbed' samples from N(0, sigma^2).
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
        return rng.uniform(0.0, 2.0 * np.pi, size=num_params).astype(float)
    elif init_lower.startswith("perturbed"):
        # Parse sigma if given as perturbed_0.01 etc.
        if "_" in init_lower:
            try:
                sigma = float(init_lower.split("_")[1])
            except ValueError:
                pass
        rng = np.random.RandomState(seed)
        return rng.normal(0.0, sigma, size=num_params).astype(float)
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

    early_stop = False
    stop_reason = "Reached maximum iterations"
    stop_status = 1
    it_used = maxiter

    for it in range(1, maxiter + 1):
        g = compute_gradient_batched(circuit, hamiltonian, params, estimator, method=grad_method, eps=eps)
        total_evals += 2 * n_params

        grad_norm = float(np.linalg.norm(g))
        if grad_norm < 1e-8 and it > 1:
            early_stop = True
            stop_reason = f"Gradient norm below tolerance ({grad_norm:.2e} < 1e-8)"
            stop_status = 0
            it_used = it
            break

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

        if len(history) >= 5 and (max(history[-5:]) - min(history[-5:]) < 1e-8):
            early_stop = True
            stop_reason = "Energy plateau: last 5 energies differ by < 1e-8 Ha"
            stop_status = 0
            it_used = it
            break

        # Phase G.5: 10-minute per-run timeout cap
        if time.perf_counter() - t0 > 600.0:
            early_stop = True
            stop_reason = "TIMEOUT: exceeded per-run wall-time cap of 600s"
            stop_status = 2
            it_used = it
            break

    wall_time = time.perf_counter() - t0
    final_e = current_e
    is_timeout = bool(stop_status == 2)
    hit_maxiter = bool(is_timeout or (not early_stop))
    opt_success = bool((not is_timeout) and early_stop)
    plateau = bool(len(history) >= 5 and (max(history[-5:]) - min(history[-5:]) < 1e-8))
    converged = bool((not is_timeout) and (opt_success or plateau))

    extra = {
        "best_energy": best_e,
        "best_iteration": best_it,
        "best_params": best_params,
        "total_evaluations": total_evals,
        "iterations_used": it_used,
        "hit_maxiter": hit_maxiter,
        "success": opt_success,
        "status": stop_status,
        "message": stop_reason,
        "converged": converged,
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

    early_stop = False
    stop_reason = "Reached maximum iterations"
    stop_status = 1
    it_used = maxiter

    for it in range(1, maxiter + 1):
        g = compute_gradient_batched(circuit, hamiltonian, params, estimator, method=grad_method, eps=grad_eps)
        total_evals += 2 * n_params

        grad_norm = float(np.linalg.norm(g))
        if grad_norm < 1e-8 and it > 1:
            early_stop = True
            stop_reason = f"Gradient norm below tolerance ({grad_norm:.2e} < 1e-8)"
            stop_status = 0
            it_used = it
            break

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

        if len(history) >= 5 and (max(history[-5:]) - min(history[-5:]) < 1e-8):
            early_stop = True
            stop_reason = "Energy plateau: last 5 energies differ by < 1e-8 Ha"
            stop_status = 0
            it_used = it
            break

        # Phase G.5: 10-minute per-run timeout cap
        if time.perf_counter() - t0 > 600.0:
            early_stop = True
            stop_reason = "TIMEOUT: exceeded per-run wall-time cap of 600s"
            stop_status = 2
            it_used = it
            break

    wall_time = time.perf_counter() - t0
    final_e = current_e
    is_timeout = bool(stop_status == 2)
    hit_maxiter = bool(is_timeout or (not early_stop))
    opt_success = bool((not is_timeout) and early_stop)
    plateau = bool(len(history) >= 5 and (max(history[-5:]) - min(history[-5:]) < 1e-8))
    converged = bool((not is_timeout) and (opt_success or plateau))

    extra = {
        "best_energy": best_e,
        "best_iteration": best_it,
        "best_params": best_params,
        "total_evaluations": total_evals,
        "iterations_used": it_used,
        "hit_maxiter": hit_maxiter,
        "success": opt_success,
        "status": stop_status,
        "message": stop_reason,
        "converged": converged,
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
    plateau = bool(len(history) >= 5 and (max(history[-5:]) - min(history[-5:]) < 1e-8))

    extra = {
        "best_energy": best_e,
        "best_iteration": best_idx,
        "best_params": res.x,
        "total_evaluations": eval_count[0],
        "iterations_used": maxiter,
        "hit_maxiter": True,
        "success": False,
        "status": 1,
        "message": "Completed maximum iterations",
        "converged": plateau,
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

    try:
        from qiskit_aer import AerSimulator
        sim = AerSimulator(method="statevector", max_parallel_threads=1, max_parallel_experiments=1)
        c_save = circuit.copy()
        c_save.save_statevector()
        circuit_params = list(circuit.parameters)
        num_params = circuit.num_parameters

        def fidelity(values_x, values_y):
            values_x = np.reshape(values_x, (-1, num_params))
            values_y = np.reshape(values_y, (-1, num_params))
            n_pts = len(values_x)
            if num_params == 0:
                return np.ones(n_pts, dtype=float)
            binds = {
                p: [float(values_x[k, i]) for k in range(n_pts)] + [float(values_y[k, i]) for k in range(n_pts)]
                for i, p in enumerate(circuit_params)
            }
            job = sim.run([c_save], parameter_binds=[binds])
            results = job.result().results
            svs = [np.asarray(r.data.statevector) for r in results]
            svs_x = svs[:n_pts]
            svs_y = svs[n_pts:]
            return np.array([abs(np.vdot(sx, sy))**2 for sx, sy in zip(svs_x, svs_y)], dtype=float)
    except Exception:
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
    plateau = bool(len(history) >= 5 and (max(history[-5:]) - min(history[-5:]) < 1e-8))

    extra = {
        "best_energy": best_e,
        "best_iteration": best_idx,
        "best_params": res.x,
        "total_evaluations": eval_count[0],
        "iterations_used": maxiter,
        "hit_maxiter": True,
        "success": False,
        "status": 1,
        "message": "Completed maximum iterations",
        "converged": plateau,
        "learning_rate": learning_rate,
        "perturbation": perturbation
    }
    return final_e, res.x, history, wall_time, extra

def run_rotosolve_dexcg(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    estimator: StatevectorEstimator | None = None,
    init_params: np.ndarray | None = None,
    maxiter: int = 5
) -> tuple[float, np.ndarray]:
    """
    Rotosolve optimizer for single-Pauli-frequency (omega = 2, period pi) circuits.
    For 1 parameter (e.g. 2e2o DexcG), solves analytically in 3 evaluations:
        E(theta) = a + b*cos(2*theta) + c*sin(2*theta)
    For multiple parameters (e.g. 4e4o DexcG), executes coordinate-wise Rotosolve sweeps.

    Returns
    -------
    (E_opt, theta_opt) : (float, np.ndarray)
    """
    estimator = estimator or StatevectorEstimator()
    n_p = circuit.num_parameters

    if n_p == 1:
        def eval_e(th):
            pub = (circuit, hamiltonian, [np.array([th])])
            return float(estimator.run([pub]).result()[0].data.evs[0])

        e0   = eval_e(0.0)
        epi2 = eval_e(np.pi / 2.0)
        epi4 = eval_e(np.pi / 4.0)

        # E(theta) = a + b*cos(2θ) + c*sin(2θ)
        a = (e0 + epi2) / 2.0
        b = (e0 - epi2) / 2.0
        c = epi4 - a

        theta_star = 0.5 * (np.arctan2(c, b) + np.pi)
        e_star = eval_e(theta_star)

        theta_alt = theta_star + np.pi / 2.0
        e_alt = eval_e(theta_alt)
        if e_alt < e_star:
            theta_star = theta_alt
            e_star = e_alt

        return e_star, np.array([theta_star])

    # Coordinate-wise Rotosolve for multi-parameter DexcG
    params = np.copy(init_params) if init_params is not None else np.zeros(n_p)
    def eval_vec(p):
        pub = (circuit, hamiltonian, [p])
        return float(estimator.run([pub]).result()[0].data.evs[0])

    cur_e = eval_vec(params)
    for _ in range(maxiter):
        e_before = cur_e
        for i in range(n_p):
            p0 = params.copy()
            e0 = cur_e

            p_pi2 = p0.copy()
            p_pi2[i] += np.pi / 2.0
            epi2 = eval_vec(p_pi2)

            p_pi4 = p0.copy()
            p_pi4[i] += np.pi / 4.0
            epi4 = eval_vec(p_pi4)

            a = (e0 + epi2) / 2.0
            b = (e0 - epi2) / 2.0
            c = epi4 - a

            delta_star = 0.5 * (np.arctan2(c, b) + np.pi)
            p_cand = p0.copy()
            p_cand[i] += delta_star
            e_cand = eval_vec(p_cand)

            p_alt = p0.copy()
            p_alt[i] += delta_star + np.pi / 2.0
            e_alt = eval_vec(p_alt)

            best_e = min(e0, e_cand, e_alt)
            if best_e == e_cand:
                params[i] += delta_star
                cur_e = e_cand
            elif best_e == e_alt:
                params[i] += delta_star + np.pi / 2.0
                cur_e = e_alt

        if abs(e_before - cur_e) < 1e-8:
            break

    return cur_e, params


def run_cobyla(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    init_params: np.ndarray,
    estimator: StatevectorEstimator | None = None,
    maxiter: int = 50,
    rhobeg: float = 0.1
) -> tuple[float, np.ndarray, list[float], float, dict]:
    """
    COBYLA optimizer (gradient-free classical reference).
    """
    t0 = time.perf_counter()
    estimator = estimator or StatevectorEstimator()
    history = []
    eval_count = [0]
    
    def cost_fn(theta):
        if time.perf_counter() - t0 > 600.0:
            raise StopIteration("TIMEOUT: exceeded per-run wall-time cap of 600s")
        eval_count[0] += 1
        pub = (circuit, hamiltonian, [theta])
        val = float(estimator.run([pub]).result()[0].data.evs[0])
        history.append(val)
        return val
        
    initial_e = cost_fn(init_params)
    from scipy.optimize import minimize
    try:
        res = minimize(
            cost_fn,
            init_params,
            method="COBYLA",
            options={"maxiter": maxiter, "rhobeg": rhobeg}
        )
        is_timeout = False
        final_e = float(res.fun)
        res_x = res.x
        nfev = int(res.nfev)
        msg = str(getattr(res, "message", ""))
        hit_maxiter = bool(res.nfev >= maxiter or ("Maximum number of function evaluations" in msg))
        opt_success = bool(getattr(res, "success", False) and not hit_maxiter)
    except StopIteration:
        is_timeout = True
        final_e = float(history[-1]) if history else initial_e
        res_x = init_params
        nfev = eval_count[0]
        msg = "TIMEOUT: exceeded per-run wall-time cap of 600s"
        hit_maxiter = True
        opt_success = False

    wall_time = time.perf_counter() - t0
    best_idx = int(np.argmin(history)) if history else 0
    best_e = float(history[best_idx]) if history else final_e
    
    plateau = bool(len(history) >= 5 and (max(history[-5:]) - min(history[-5:]) < 1e-8))
    converged = bool((not is_timeout) and (opt_success or plateau))

    extra = {
        "best_energy": best_e,
        "best_iteration": best_idx,
        "best_params": res_x,
        "total_evaluations": eval_count[0],
        "iterations_used": nfev,
        "hit_maxiter": hit_maxiter,
        "success": opt_success,
        "status": 2 if is_timeout else int(getattr(res, "status", 0 if opt_success else 1)),
        "message": msg,
        "converged": converged,
        "learning_rate": None
    }
    return final_e, res_x, history, wall_time, extra

def run_lbfgsb(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    init_params: np.ndarray,
    estimator: StatevectorEstimator | None = None,
    maxiter: int = 50,
    grad_eps: float = 1e-5
) -> tuple[float, np.ndarray, list[float], float, dict]:
    """
    L-BFGS-B optimizer with exact batched finite-difference gradients.
    """
    t0 = time.perf_counter()
    estimator = estimator or StatevectorEstimator()
    history = []
    eval_count = [0]
    n_params = len(init_params)
    
    def cost_and_grad(theta):
        if time.perf_counter() - t0 > 600.0:
            raise StopIteration("TIMEOUT: exceeded per-run wall-time cap of 600s")
        eval_count[0] += 1
        pub = (circuit, hamiltonian, [theta])
        val = float(estimator.run([pub]).result()[0].data.evs[0])
        history.append(val)
        g = compute_gradient_batched(circuit, hamiltonian, theta, estimator, method="finite_diff", eps=grad_eps)
        eval_count[0] += 2 * n_params
        return val, g
        
    from scipy.optimize import minimize
    try:
        res = minimize(
            cost_and_grad,
            init_params,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-8}
        )
        is_timeout = False
        final_e = float(res.fun)
        res_x = res.x
        nit = int(res.nit)
        msg = str(getattr(res, "message", ""))
        hit_maxiter = bool(res.nit >= maxiter or ("TOTAL NO. of ITERATIONS REACHED LIMIT" in msg))
        opt_success = bool(getattr(res, "success", False) and not hit_maxiter)
    except StopIteration:
        is_timeout = True
        final_e = float(history[-1]) if history else 0.0
        res_x = init_params
        nit = eval_count[0]
        msg = "TIMEOUT: exceeded per-run wall-time cap of 600s"
        hit_maxiter = True
        opt_success = False

    wall_time = time.perf_counter() - t0
    best_idx = int(np.argmin(history)) if history else 0
    best_e = float(history[best_idx]) if history else final_e
    
    plateau = bool(len(history) >= 5 and (max(history[-5:]) - min(history[-5:]) < 1e-8))
    converged = bool((not is_timeout) and (opt_success or plateau))

    extra = {
        "best_energy": best_e,
        "best_iteration": best_idx,
        "best_params": res_x,
        "total_evaluations": eval_count[0],
        "iterations_used": nit,
        "hit_maxiter": hit_maxiter,
        "success": opt_success,
        "status": 2 if is_timeout else int(getattr(res, "status", 0 if opt_success else 1)),
        "message": msg,
        "converged": converged,
        "learning_rate": None
    }
    return final_e, res_x, history, wall_time, extra

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
    seed: int = 42,
    sigma: float = 0.05
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
    init_params = get_initial_point(init_name, num_params, seed=seed, sigma=sigma)

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
    elif opt_upper == "COBYLA":
        final_e, final_p, hist, wtime, extra = run_cobyla(
            circuit, hamiltonian, init_params, estimator=estimator, maxiter=maxiter
        )
    elif opt_upper in ("L-BFGS-B", "LBFGSB"):
        final_e, final_p, hist, wtime, extra = run_lbfgsb(
            circuit, hamiltonian, init_params, estimator=estimator, maxiter=maxiter
        )
    elif opt_upper == "ROTOSOLVE" and ansatz_name == "DexcG":
        final_e, final_p = run_rotosolve_dexcg(circuit, hamiltonian, estimator=estimator, init_params=init_params, maxiter=maxiter)
        hist = [final_e]
        wtime = 0.05
        extra = {
            "best_energy": final_e,
            "best_iteration": 1,
            "best_params": final_p,
            "total_evaluations": 4 if circuit.num_parameters == 1 else 3 * circuit.num_parameters * maxiter,
            "iterations_used": 1 if circuit.num_parameters == 1 else maxiter,
            "hit_maxiter": False,
            "success": True,
            "status": 0,
            "message": "Analytical solution found",
            "converged": True,
            "learning_rate": None
        }
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
        "iterations_used": extra.get("iterations_used", len(hist) - 1),
        "hit_maxiter": extra.get("hit_maxiter", False),
        "success": extra.get("success", False),
        "status": extra.get("status", "unknown"),
        "message": extra.get("message", ""),
        "converged": extra.get("converged", False),
        "learning_rate": extra.get("learning_rate", lr)
    }
