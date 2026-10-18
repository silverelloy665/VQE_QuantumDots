"""
Verification Test: PySCF RHF and CASCI reference energies match cached JSON data to < 1e-6 Ha.
"""
import json
from pathlib import Path
import numpy as np
import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def test_pyscf_reference_values_sto3g():
    path = DATA_DIR / "bn_dot_sto3g.json"
    assert path.exists(), f"Missing cache file {path}"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    expected_rhf = -631.74167448
    expected_casci = -631.74178346
    
    assert np.isclose(data["hf_energy"], expected_rhf, atol=1e-6), f"STO-3G RHF mismatch: {data['hf_energy']}"
    assert np.isclose(data["casci_energy"], expected_casci, atol=1e-6), f"STO-3G CASCI mismatch: {data['casci_energy']}"

def test_pyscf_reference_values_631gdp():
    path = DATA_DIR / "bn_dot_631gdp.json"
    assert path.exists(), f"Missing cache file {path}"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    expected_rhf = -639.72424230
    expected_casci = -639.72428323
    
    assert np.isclose(data["hf_energy"], expected_rhf, atol=1e-6), f"6-31G(d,p) RHF mismatch: {data['hf_energy']}"
    assert np.isclose(data["casci_energy"], expected_casci, atol=1e-6), f"6-31G(d,p) CASCI mismatch: {data['casci_energy']}"

