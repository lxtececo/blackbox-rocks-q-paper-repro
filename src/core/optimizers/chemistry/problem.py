"""Electronic-structure problem setup and the classical reference energies.

PySCF does the textbook part: a Hartree-Fock (mean-field) calculation gives
the molecular orbitals, the one- and two-electron integrals in that basis,
and the HF energy. Full configuration interaction (FCI) then diagonalises the
Hamiltonian exactly in the same basis, which is affordable only because the
presets are small. FCI is the ground truth every run is scored against.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.optimizers.chemistry.molecules import BASIS, MOLECULES, geometry_for, resolve_bond_length

HARTREE_TO_KCAL = 627.509474
HARTREE_TO_EV = 27.211386
#: 1 kcal/mol, the conventional "chemical accuracy" threshold.
CHEMICAL_ACCURACY_HA = 1.0 / HARTREE_TO_KCAL


@dataclass
class MolecularProblem:
    molecule_id: str
    bond_length: Optional[float]
    geometry: str
    norb: int
    nelec: Tuple[int, int]
    h1e: np.ndarray
    h2e: np.ndarray
    ecore: float
    hf_energy: float
    fci_energy: float
    fci_occupancies: Tuple[List[float], List[float]]
    setup_seconds: float

    @property
    def n_qubits(self) -> int:
        return 2 * self.norb

    @property
    def fci_dimension(self) -> int:
        return math.comb(self.norb, self.nelec[0]) * math.comb(self.norb, self.nelec[1])

    @property
    def correlation_energy(self) -> float:
        return self.fci_energy - self.hf_energy

    def describe(self) -> Dict[str, Any]:
        spec = MOLECULES[self.molecule_id]
        return {
            "id": self.molecule_id,
            "label": spec["label"],
            "formula": spec["formula"],
            "bond_length": self.bond_length,
            "geometry": self.geometry,
            "basis": BASIS,
            "n_orbitals": self.norb,
            "n_electrons": list(self.nelec),
            "n_qubits": self.n_qubits,
        }

    def reference(self) -> Dict[str, Any]:
        return {
            "hf_energy": round(self.hf_energy, 8),
            "fci_energy": round(self.fci_energy, 8),
            "correlation_energy": round(self.correlation_energy, 8),
            "nuclear_repulsion": round(self.ecore, 8),
            "fci_dimension": self.fci_dimension,
            "chemical_accuracy_ha": round(CHEMICAL_ACCURACY_HA, 6),
            "fci_occupancies": {
                "alpha": [round(x, 5) for x in self.fci_occupancies[0]],
                "beta": [round(x, 5) for x in self.fci_occupancies[1]],
            },
        }


def build_problem(molecule_id: str, bond_length: Optional[float] = None) -> MolecularProblem:
    """Run HF + FCI for one preset and return the integrals and references."""
    from pyscf import ao2mo, fci, gto, scf  # noqa: PLC0415

    if molecule_id not in MOLECULES:
        raise ValueError(f"unknown molecule '{molecule_id}'")
    started = time.monotonic()
    length = resolve_bond_length(molecule_id, bond_length)
    geometry = geometry_for(molecule_id, length)

    mol = gto.M(atom=geometry, basis=BASIS, symmetry=False, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()
    if not mf.converged:
        # Stretched bonds can trip plain SCF; second-order SCF is more robust.
        mf = scf.RHF(mol).newton()
        mf.conv_tol = 1e-10
        mf.kernel()

    mo = mf.mo_coeff
    norb = mo.shape[1]
    h1e = mo.T @ mf.get_hcore() @ mo
    h2e = ao2mo.restore(1, ao2mo.full(mol, mo), norb)
    n_alpha, n_beta = mol.nelec

    solver = fci.direct_spin1.FCI()
    solver.max_cycle = 200
    solver.conv_tol = 1e-12
    fci_energy, civec = solver.kernel(h1e, h2e, norb, (n_alpha, n_beta), ecore=mol.energy_nuc())
    dm_a, dm_b = solver.make_rdm1s(civec, norb, (n_alpha, n_beta))

    return MolecularProblem(
        molecule_id=molecule_id,
        bond_length=length,
        geometry=geometry,
        norb=int(norb),
        nelec=(int(n_alpha), int(n_beta)),
        h1e=np.asarray(h1e),
        h2e=np.asarray(h2e),
        ecore=float(mol.energy_nuc()),
        hf_energy=float(mf.e_tot),
        fci_energy=float(fci_energy),
        fci_occupancies=(
            [float(x) for x in np.diag(dm_a)],
            [float(x) for x in np.diag(dm_b)],
        ),
        setup_seconds=time.monotonic() - started,
    )


def score_energy(energy: float, problem: MolecularProblem) -> Dict[str, Any]:
    """Error vs exact, in the units chemists quote, plus correlation recovered."""
    error = energy - problem.fci_energy
    correlation = problem.correlation_energy
    recovered = None
    if abs(correlation) > 1e-9:
        recovered = 100.0 * (problem.hf_energy - energy) / (problem.hf_energy - problem.fci_energy)
    return {
        "energy": round(energy, 8),
        "error_vs_fci": round(error, 8),
        "error_mha": round(error * 1000.0, 4),
        "error_kcal_mol": round(error * HARTREE_TO_KCAL, 4),
        "within_chemical_accuracy": abs(error) <= CHEMICAL_ACCURACY_HA,
        "correlation_recovered_pct": round(recovered, 2) if recovered is not None else None,
    }
