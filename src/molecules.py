import os

import numpy as np
from pyscf import gto, mcscf, scf
from pyscf.fci import cistring, direct_spin1
from scipy.linalg import eigh

import math

# Stretched H4 has several determinants that matter, while LiH and water sit close
# to a single Hartree-Fock determinant, so the set covers both regimes
MOLECULES = {
    "h4": dict(atom="H 0 0 0; H 0 0 1.5; H 0 0 3.0; H 0 0 4.5", basis="6-31g"),
    "lih": dict(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g"),
    # TODO add water as "h2o" in sto-3g, any reasonable geometry works
}


def integrals(name: str):
    """RHF, then one- and two-electron integrals over every orbital"""

    mol = gto.M(**MOLECULES[name], verbose=0)
    mf = scf.RHF(mol).run()
    mc = mcscf.CASCI(mf, mol.nao, mol.nelec)
    h1, ecore = mc.get_h1eff()
    return h1, mc.get_h2eff(), ecore, mol.nao, mol.nelec, mf.e_tot


def determinant_labels(norb: int, nelec: tuple[int, int]) -> np.ndarray:
    """One (alpha, beta) bit-mask pair per determinant, in PySCF's FCI vector order"""

    labels = np.zeros((math.comb(norb, nelec[0])*math.comb(norb, nelec[1]), 2))

    up = 2**nelec[0]-1
    down = 2**nelec[1]-1

    def nextstring(bitstring: int, spaces: int):  # gets the next label according to the ordering
        if bitstring == 0:
            return False, False
        maxind = np.floor(np.log2(bitstring))
        if maxind == spaces-1:
            partstring, newmaxind = nextstring(bitstring-2**maxind, spaces-1)
            if partstring == False:
                return False, False
            return partstring + 2**(newmaxind+1), newmaxind+1
        else:
            return bitstring+2**maxind, maxind+1
        
        
    for i in range(labels.shape[0]):
        labels[i][0] = up
        labels[i][1] = down
        down, _ = nextstring(down, norb)
        if down == False:
            down = 2**nelec[1]-1
            up, _ = nextstring(up, norb)

    return labels


def fci_hamiltonian(h1, h2, ecore, norb: int, nelec: tuple[int, int]) -> np.ndarray:
    """Full CI Hamiltonian in the determinant basis, core energy included"""

    n_alpha = cistring.num_strings(norb, nelec[0])
    n_dets = n_alpha * cistring.num_strings(norb, nelec[1])

    # pspace builds H over the lowest-diagonal determinants for PySCF's FCI
    # preconditioner. Asking for all of them gives the whole matrix in its own order
    addr, h_sub = direct_spin1.pspace(h1, h2, norb, nelec, np=n_dets)
    H = np.zeros((n_dets, n_dets))
    H[np.ix_(addr, addr)] = h_sub
    return H + ecore * np.eye(n_dets)


def main(results_dir: str, molecule: str, overwrite: bool):

    out_path = f"{results_dir}/hamiltonians/{molecule}.npz"
    if os.path.exists(out_path) and not overwrite:
        print(f"Skipping {out_path}, already done")
        return

    h1, h2, ecore, norb, nelec, e_hf = integrals(molecule)
    H = fci_hamiltonian(h1, h2, ecore, norb, nelec)
    labels = determinant_labels(norb, nelec)
    assert len(labels) == len(H), "one label per determinant"
    assert np.allclose(H, H.T), "hamiltonian is not symmetric"
    print(f"{molecule}: {norb} orbitals, {nelec} electrons, H has a shape of {H.shape}")

    E, C = eigh(H)
    print(f"Hartree-Fock {e_hf:.6f}, full CI {E[0]:.6f} hartree")
    print(f"Correlation energy {E[0] - e_hf:.6f} hartree")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, H=H, labels=labels, E=E, c0=C[:, 0], units="hartree")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("-m", "--molecule", type=str, default="h4")
    parser.add_argument("--overwrite", action="store_true")
    _args = parser.parse_args()
    main(**vars(_args))
