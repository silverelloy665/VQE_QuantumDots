import os
from pathlib import Path
from dotenv import load_dotenv
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.providers.fake_provider import GenericBackendV2

# Single source of truth for target hardware backend
DEFAULT_HARDWARE_BACKEND = "ibm_marrakesh"

# Load environment variables from .env if present
ROOT_DIR = Path(__file__).resolve().parent.parent
dotenv_path = ROOT_DIR / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)

def get_runtime_service(
    token: str | None = None,
    channel: str | None = None,
    instance: str | None = None
) -> QiskitRuntimeService:
    """
    Initializes and returns an instance of QiskitRuntimeService.
    Falls back to environment variables or saved local disk credentials.
    """
    token = (
        token
        or os.getenv("IBM_QUANTUM_TOKEN")
        or os.getenv("IBMQ_API_KEY")
        or os.getenv("IBM_QUANTUM_API_KEY")
        or os.getenv("QISKIT_IBM_TOKEN")
    )
    channel = channel or os.getenv("IBM_QUANTUM_CHANNEL", "ibm_quantum_platform")
    
    try:
        if token:
            return QiskitRuntimeService(channel=channel, token=token, instance=instance)
        else:
            return QiskitRuntimeService()
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize QiskitRuntimeService. Ensure your API key is configured. Error: {e}"
        )


def get_target_backend(backend_name: str | None = None, service: QiskitRuntimeService | None = None):
    """
    Resolves the canonical backend (default: DEFAULT_HARDWARE_BACKEND).
    Falls back to a GenericBackendV2 with 156 qubits and Heron-like gates
    if offline or credentials are not configured.
    """
    target = backend_name or DEFAULT_HARDWARE_BACKEND
    try:
        service = service or get_runtime_service()
        backend = service.backend(target)
        return backend
    except Exception:
        # Fallback mock with same qubit count (156) and native gate set (ECR / CZ)
        return GenericBackendV2(num_qubits=156, basis_gates=["ecr", "cz", "id", "rz", "sx", "x"])


if __name__ == "__main__":
    service = get_runtime_service()
    backends = [b.name for b in service.backends()]
    print(f"[OK] IBM Quantum Service initialized successfully!")
    print(f"Default hardware backend: {DEFAULT_HARDWARE_BACKEND}")
    print(f"Available backends: {backends}")
