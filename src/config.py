import os
from pathlib import Path
from dotenv import load_dotenv
from qiskit_ibm_runtime import QiskitRuntimeService

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

if __name__ == "__main__":
    service = get_runtime_service()
    backends = [b.name for b in service.backends()]
    print(f"[OK] IBM Quantum Service initialized successfully!")
    print(f"Available backends: {backends}")
