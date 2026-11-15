"""Molecular Energy vertical (chemistry): ground-state energy of small
molecules by SqDRIFT, scored against the exact classical answer.

* ``molecules`` - presets a non-chemist can pick from.
* ``problem``   - Hartree-Fock + exact (full CI) reference energies.
* ``sqdrift``   - qDRIFT circuit ensemble + sample-based diagonalization.
* ``service``   - orchestration used by ``api/chemistry_endpoints.py``.
"""
