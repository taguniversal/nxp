# viz_ca.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

OCC, SPEC, Q = 0, 1, 2
BXP, BXM, BYP, BYM, BZP, BZM = 3, 4, 5, 6, 7, 8
PHI = 9

def extract_atoms(grid):
    occ = grid[0, OCC]
    spec = grid[0, SPEC]
    xs, ys, zs = np.where(occ == 1)
    species = spec[xs, ys, zs]
    atoms = np.stack([xs, ys, zs, species], axis=1)
    return atoms

def extract_bonds(grid):
    occ = grid[0, OCC]
    bonds = []
    X, Y, Z = occ.shape
    # +X
    bxp = grid[0, BXP]
    x, y, z = np.where(bxp == 1)
    for xi, yi, zi in zip(x, y, z):
        x2 = (xi + 1) % X
        if occ[xi, yi, zi] and occ[x2, yi, zi]:
            bonds.append(((xi, yi, zi), (x2, yi, zi)))
    # +Y
    byp = grid[0, BYP]
    x, y, z = np.where(byp == 1)
    for xi, yi, zi in zip(x, y, z):
        y2 = (yi + 1) % Y
        if occ[xi, yi, zi] and occ[xi, y2, zi]:
            bonds.append(((xi, yi, zi), (xi, y2, zi)))
    # +Z
    bzp = grid[0, BZP]
    x, y, z = np.where(bzp == 1)
    for xi, yi, zi in zip(x, y, z):
        z2 = (zi + 1) % Z
        if occ[xi, yi, zi] and occ[xi, yi, z2]:
            bonds.append(((xi, yi, zi), (xi, yi, z2)))
    return bonds

def plot_atoms_bonds(atoms, bonds, title="CA Molecular Snapshot", save_path=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if len(atoms) > 0:
        xs, ys, zs = atoms[:,0], atoms[:,1], atoms[:,2]
        ax.scatter(xs, ys, zs, s=60)

    for (a, b) in bonds:
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=35)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160, bbox_inches='tight')
    plt.show()

# Example usage with your sim grid:
# from kernel_test import ca_step_jit, empty_grid, place_atom, current_bonds
# import jax.numpy as jnp
# g = empty_grid(1, 16, 16, 16)
# g = place_atom(g, 0, 8, 8, 8, species=6, phi=10)
# g = place_atom(g, 0, 9, 8, 8, species=8, phi=9)
# for _ in range(5):
#     g = ca_step_jit(g)
# # JAX → NumPy
# g_np = np.array(g)
# atoms = extract_atoms(g_np)
# bonds = extract_bonds(g_np)
# plot_atoms_bonds(atoms, bonds, title="Two-Atom Bond")
