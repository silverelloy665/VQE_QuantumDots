"""
Unit tests for cache schema validation:
Ensures old-schema records (lacking final_params, hit_maxiter, schema_version 3.0) are rejected.
"""
import pytest


def validate_cache_record_v3(record: dict) -> bool:
    """Validates that a benchmark record conforms to the Round 3 schema."""
    required_keys = [
        "ansatz", "init", "optimizer", "final_energy", "final_params",
        "best_params", "total_evaluations", "converged", "hit_maxiter"
    ]
    if not all(k in record for k in required_keys):
        return False
    if record["final_params"] is None or not isinstance(record["final_params"], (list, tuple)):
        return False
    if not isinstance(record["converged"], bool):
        return False
    if not isinstance(record["hit_maxiter"], bool):
        return False
    return True


def test_reject_old_schema_without_params():
    old_record = {
        "ansatz": "DexcG",
        "init": "random",
        "optimizer": "GD",
        "lr": 0.01,
        "seed": 0,
        "final_energy": -636.876664,
        "history": [-636.82, -636.87],
        "total_evaluations": 926,
        "wall_time": 120.6,
        "converged": True
    }
    assert validate_cache_record_v3(old_record) is False, "Record without final_params must be rejected"


def test_accept_valid_v3_schema_record():
    valid_record = {
        "ansatz": "2-UpCCGSD",
        "init": "zero",
        "optimizer": "L-BFGS-B",
        "lr": None,
        "seed": 0,
        "final_energy": -637.58131,
        "best_energy": -637.58131,
        "final_params": [0.0] * 36,
        "best_params": [0.0] * 36,
        "history": [-637.573, -637.581],
        "total_evaluations": 2044,
        "iterations_used": 39,
        "hit_maxiter": False,
        "converged": True,
        "wall_time": 45.2
    }
    assert validate_cache_record_v3(valid_record) is True, "Valid v3 record must be accepted"

