"""
Optimizer wrappers and execution harness for VQE benchmark.
Includes: GD (Gradient Descent), SPSA, ADAM, QNSPSA with V2 primitives.
Uses vectorized batched PUB evaluations for fast exact gradient computation.
"""
import time
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit_algorithms.optimizers import SPSA, QNSPSA

def get_initial_point(init_name: str, num_params: int, seed: int = 42) -> np.ndarray:
    """
    Returns initial parameter vector for the specified initialization strategy.
    Options: 'zero', 'half', 'one', 'random'.
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
        return rng.uniform(0.0, 1.0, size=num_params).astype(float)
    else:
        raise ValueError(f"Unknown initialization strategy: {init_name}")

def compute_gradient_batched(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    params: np.ndarray,
    estimator: StatevectorEstimator,
    method: str = "param_shift",
    shift: float = np.pi / 2.0,
    eps: float = 1e-4
) -> np.ndarray:
    """
    Vectorized batched gradient computation.
    Evaluates all 2N parameter points in a single PUB call.
    """
    n = len(params)
    if n == 0:
        return np.array([], dtype=float)
        
    delta = shift if method == "param_shift" else eps
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
    denom = (2.0 * np.sin(shift)) if method == "param_shift" else (2.0 * eps)
    for i in range(n):
        grad[i] = (evs[2 * i] - evs[2 * i + 1]) / denom
        
    return grad

def run_gradient_descent(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    init_params: np.ndarray,
    estimator: StatevectorEstimator,
    maxiter: int = 50,
    lr: float = 0.05,
    grad_method: str = "param_shift"
) -> tuple[float, np.ndarray, list[float], float]:
    """
    Standard Gradient Descent: θ_{t} = θ_{t-1} - lr * ∇E(θ_{t-1})
    """
    t0 = time.perf_counter()
    params = np.copy(init_params)
    history = []
    
    pub_single = (circuit, hamiltonian, [params])
    current_e = float(estimator.run([pub_single]).result()[0].data.evs[0])
    
    for it in range(maxiter):
        history.append(current_e)
        g = compute_gradient_batched(circuit, hamiltonian, params, estimator, method=grad_method)
        if np.all(np.abs(g) < 1e-12) and it == 0 and not np.all(params == 0):
            g = compute_gradient_batched(circuit, hamiltonian, params, estimator, method="finite_diff")
        params = params - lr * g
        current_e = float(estimator.run([(circuit, hamiltonian, [params])]).result()[0].data.evs[0])
        
    wall_time = time.perf_counter() - t0
    final_e = current_e
    return final_e, params, history, wall_time

def run_adam(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    init_params: np.ndarray,
    estimator: StatevectorEstimator,
    maxiter: int = 50,
    lr: float = 0.05,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    grad_method: str = "param_shift"
) -> tuple[float, np.ndarray, list[float], float]:
    """
    ADAM Optimizer with vectorized batched gradients.
    """
    t0 = time.perf_counter()
    params = np.copy(init_params)
    history = []
    
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    
    current_e = float(estimator.run([(circuit, hamiltonian, [params])]).result()[0].data.evs[0])
    
    for it in range(1, maxiter + 1):
        history.append(current_e)
        g = compute_gradient_batched(circuit, hamiltonian, params, estimator, method=grad_method)
        if np.all(np.abs(g) < 1e-12) and it == 1 and not np.all(params == 0):
            g = compute_gradient_batched(circuit, hamiltonian, params, estimator, method="finite_diff")
            
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g ** 2)
        
        m_hat = m / (1.0 - beta1 ** it)
        v_hat = v / (1.0 - beta2 ** it)
        
        params = params - lr * m_hat / (np.sqrt(v_hat) + eps)
        current_e = float(estimator.run([(circuit, hamiltonian, [params])]).result()[0].data.evs[0])
        
    wall_time = time.perf_counter() - t0
    final_e = current_e
    return final_e, params, history, wall_time

def run_spsa(
    energy_fn,
    init_params: np.ndarray,
    maxiter: int = 50,
    learning_rate: float = 0.1,
    perturbation: float = 0.1,
    seed: int = 42
) -> tuple[float, np.ndarray, list[float], float]:
    """
    SPSA Optimizer with fixed learning rate and perturbation.
    """
    t0 = time.perf_counter()
    history = []
    
    def callback(nfev, x, fx, stepsize, accepted):
        history.append(float(fx))
        
    spsa = SPSA(
        maxiter=maxiter,
        learning_rate=learning_rate,
        perturbation=perturbation,
        callback=callback
    )
    
    res = spsa.minimize(energy_fn, init_params)
    wall_time = time.perf_counter() - t0
    
    while len(history) < maxiter:
        history.append(float(res.fun))
    if len(history) > maxiter:
        history = history[:maxiter]
        
    return float(res.fun), res.x, history, wall_time

def run_qnspsa(
    energy_fn,
    circuit: QuantumCircuit,
    init_params: np.ndarray,
    maxiter: int = 50,
    learning_rate: float = 0.1,
    perturbation: float = 0.1,
    sampler: StatevectorSampler | None = None,
    seed: int = 42
) -> tuple[float, np.ndarray, list[float], float]:
    """
    QNSPSA with quantum state fidelity metric.
    """
    t0 = time.perf_counter()
    history = []
    sampler = sampler or StatevectorSampler()
    
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
    
    res = qnspsa.minimize(energy_fn, init_params)
    wall_time = time.perf_counter() - t0
    
    while len(history) < maxiter:
        history.append(float(res.fun))
    if len(history) > maxiter:
        history = history[:maxiter]
        
    return float(res.fun), res.x, history, wall_time

def run_vqe_single(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    ansatz_name: str,
    init_name: str,
    optimizer_name: str,
    maxiter: int = 50,
    estimator: StatevectorEstimator | None = None,
    sampler: StatevectorSampler | None = None,
    seed: int = 42
) -> dict:
    """
    Runs a single VQE configuration and returns dictionary of metrics.
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
        final_e, final_params, hist, wtime = run_gradient_descent(
            circuit, hamiltonian, init_params, estimator, maxiter=maxiter, lr=0.05
        )
    elif opt_upper == "ADAM":
        final_e, final_params, hist, wtime = run_adam(
            circuit, hamiltonian, init_params, estimator, maxiter=maxiter, lr=0.05
        )
    elif opt_upper == "SPSA":
        final_e, final_params, hist, wtime = run_spsa(
            cost_fn, init_params, maxiter=maxiter, learning_rate=0.1, perturbation=0.1, seed=seed
        )
    elif opt_upper == "QNSPSA":
        final_e, final_params, hist, wtime = run_qnspsa(
            cost_fn, circuit, init_params, maxiter=maxiter, learning_rate=0.1, perturbation=0.1, sampler=sampler, seed=seed
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
    return {
        "ansatz": ansatz_name,
        "initialization": init_name,
        "optimizer": opt_upper,
        "final_energy": final_e,
        "final_params": final_params,
        "energy_history": hist,
        "wall_time": wtime,
        "num_params": num_params
    }

