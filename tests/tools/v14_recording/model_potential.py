"""
Analytic ammonia-like potential for testing the MWNEB machinery without ORCA.

Atoms: N, H, H, H (in that order). Coordinates in bohr, energy in Eh.

    E = E0 + sum_i  K_NH (r_NH_i - R0)^2
           + sum_ij K_HH (d_HH_ij - D0)^2
           + VB * (h^2 - H0^2)^2 / H0^4

h is the signed height of N above the plane of the three H atoms, so the last
term is a symmetric double well (umbrella inversion) with a barrier VB at the
planar geometry. Every term depends only on distances and on h, so the energy is
invariant to rotations and translations, like a real electronic energy.

The gradient is a central finite difference. It is deterministic, which is all
the tests need: same geometry in, same numbers out, on any machine.
"""

import numpy as np

E0   = -56.2       # Eh, just to look like a real total energy
K_NH = 0.25        # Eh / bohr^2
R0   = 1.91        # bohr  (1.01 Angstrom)
K_HH = 0.05        # Eh / bohr^2
D0   = 3.064       # bohr
VB   = 0.0093      # Eh    (about 5.8 kcal/mol, close to the real NH3 barrier)
H0   = 0.72        # bohr  (height of N above the H3 plane at the minima)
FD_STEP = 1.0e-5   # bohr


def energy(coords):
    """coords: (4, 3) array in bohr, order N, H, H, H."""
    c = np.asarray(coords, dtype=float)
    N, H = c[0], c[1:]
    e = E0
    for i in range(3):
        e += K_NH * (np.linalg.norm(N - H[i]) - R0) ** 2
    for i in range(3):
        for j in range(i + 1, 3):
            e += K_HH * (np.linalg.norm(H[i] - H[j]) - D0) ** 2
    normal = np.cross(H[1] - H[0], H[2] - H[0])
    normal = normal / np.linalg.norm(normal)
    h = float(np.dot(normal, N - H[0]))
    e += VB * (h ** 2 - H0 ** 2) ** 2 / H0 ** 4
    return float(e)


def energy_and_gradient(coords):
    """Energy (Eh) and gradient (Eh/bohr, shape (4, 3))."""
    c = np.asarray(coords, dtype=float)
    grad = np.zeros_like(c)
    for a in range(c.shape[0]):
        for k in range(3):
            plus, minus = c.copy(), c.copy()
            plus[a, k] += FD_STEP
            minus[a, k] -= FD_STEP
            grad[a, k] = (energy(plus) - energy(minus)) / (2.0 * FD_STEP)
    return energy(c), grad


def endpoints(rotate_product=True):
    """
    Reactant (N above the H3 plane) and product (N below), in bohr.
    The product is rigidly rotated and shifted so the NEB's Kabsch alignment
    has real work to do.
    """
    rho = np.sqrt(R0 ** 2 - H0 ** 2)
    ang = np.radians([90.0, 210.0, 330.0])
    H = np.column_stack([rho * np.cos(ang), rho * np.sin(ang), np.zeros(3)])
    H[1] += [0.02, -0.01, 0.0]          # break the perfect C3v symmetry a little
    reactant = np.vstack([[0.0, 0.0, +H0], H])
    product = np.vstack([[0.0, 0.0, -H0], H])
    if rotate_product:
        th = np.radians(35.0)
        Rx = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
        product = product @ Rx.T + np.array([0.3, -0.2, 0.1])
    return reactant, product


def engine(images_cart, indices, work_dir):
    """In-process engine with the runner's engine signature (no files, no ORCA)."""
    energies, gradients = [], []
    for i in indices:
        e, g = energy_and_gradient(images_cart[i])
        energies.append(e)
        gradients.append(g)
    return np.array(energies), np.array(gradients)
