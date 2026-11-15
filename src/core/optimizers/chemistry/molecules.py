"""Molecule presets for the Molecular Energy vertical.

Each preset is a small molecule in the minimal STO-3G basis, chosen so the
exact (full CI) answer is still computable classically. That is what lets
every quantum run be scored against a known-correct energy rather than only
against another estimate.

Diatomics expose their bond length, because stretching a bond is the most
intuitive knob there is: near equilibrium the simple mean-field picture
(Hartree-Fock) is decent, and as the bond breaks it fails badly. That gap is
the "correlation energy" the quantum method is trying to recover.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

BASIS = "sto-3g"

MOLECULES: Dict[str, Dict[str, Any]] = {
    "h2": {
        "label": "Hydrogen (H\u2082)",
        "formula": "H2",
        "description": "The simplest molecule: two protons sharing two electrons. "
                       "A sanity check that runs in seconds.",
        "everyday": "Fuel cells and rocket fuel.",
        "atoms": ("H", "H"),
        "bond_length": {"default": 0.735, "min": 0.4, "max": 3.0, "stretched": 2.0},
        "n_orbitals": 2,
        "n_electrons": 2,
        "difficulty": "starter",
    },
    "lih": {
        "label": "Lithium hydride (LiH)",
        "formula": "LiH",
        "description": "A light metal bonded to hydrogen. Small, but large enough "
                       "that the quantum circuits stop being trivial.",
        "everyday": "Hydrogen storage and battery research.",
        "atoms": ("Li", "H"),
        "bond_length": {"default": 1.595, "min": 1.0, "max": 4.0, "stretched": 3.0},
        "n_orbitals": 6,
        "n_electrons": 4,
        "difficulty": "starter",
    },
    "h2o": {
        "label": "Water (H\u2082O)",
        "formula": "H2O",
        "description": "Bent molecule at its experimental geometry (0.96 \u00c5 O-H bonds, "
                       "104.5\u00b0 angle). Fixed shape, no bond control.",
        "everyday": "Everything wet.",
        "geometry": "O 0 0 0; H 0.757 0.586 0; H -0.757 0.586 0",
        "bond_length": None,
        "n_orbitals": 7,
        "n_electrons": 10,
        "difficulty": "intermediate",
    },
    "n2": {
        "label": "Nitrogen (N\u2082)",
        "formula": "N2",
        "description": "The IBM SqDRIFT tutorial molecule: a triple bond that is "
                       "notoriously hard for simple methods once stretched.",
        "everyday": "78% of the air; breaking this bond is what fertiliser plants spend "
                    "1-2% of the world's energy on.",
        "atoms": ("N", "N"),
        "bond_length": {"default": 1.09, "min": 0.9, "max": 2.5, "stretched": 1.8},
        "n_orbitals": 10,
        "n_electrons": 14,
        "difficulty": "advanced",
    },
}

DEFAULT_MOLECULE = "h2o"


def qubits_for(molecule_id: str) -> int:
    """Jordan-Wigner maps each spin orbital to one qubit."""
    return 2 * int(MOLECULES[molecule_id]["n_orbitals"])


def geometry_for(molecule_id: str, bond_length: Optional[float] = None) -> str:
    """PySCF atom string, honouring the bond length for diatomics."""
    spec = MOLECULES[molecule_id]
    if spec.get("geometry"):
        return spec["geometry"]
    bounds = spec["bond_length"]
    length = float(bond_length if bond_length is not None else bounds["default"])
    a, b = spec["atoms"]
    return f"{a} 0 0 0; {b} 0 0 {length:.4f}"


def resolve_bond_length(molecule_id: str, bond_length: Optional[float]) -> Optional[float]:
    """Clamp a requested bond length into the preset's range (None if fixed)."""
    bounds = MOLECULES[molecule_id].get("bond_length")
    if not bounds:
        return None
    if bond_length is None:
        return float(bounds["default"])
    return float(min(max(float(bond_length), bounds["min"]), bounds["max"]))


def catalog_entries() -> list[Dict[str, Any]]:
    return [
        {
            "id": key,
            "label": spec["label"],
            "formula": spec["formula"],
            "description": spec["description"],
            "everyday": spec["everyday"],
            "bond_length": spec.get("bond_length"),
            "n_orbitals": spec["n_orbitals"],
            "n_electrons": spec["n_electrons"],
            "n_qubits": qubits_for(key),
            "basis": BASIS,
            "difficulty": spec["difficulty"],
        }
        for key, spec in MOLECULES.items()
    ]
