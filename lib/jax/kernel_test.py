# kernel_test.py  (5D-safe rolls + overflow-safe hash)
# ------------------------------------------------------------
# Minimal JAX cellular-automaton kernel for NKS-style molecules
#  - 3D lattice, 6-neighbor bonds
#  - integer state channels
#  - valence-checked bond adds + deterministic conflict resolver
#  - simple scalar field diffusion (PHI)
#
# Grid shape: [B, C, X, Y, Z]
# Channels:
#   0: OCC (0/1)          uint8
#   1: SPEC (species id)  uint8
#   2: Q   (charge)       int8
#   3..8: bond bits (+X,-X,+Y,-Y,+Z,-Z)  uint8
#   9: PHI (scalar field) int16
# ------------------------------------------------------------

from typing import Tuple
import jax
import jax.numpy as jnp

# ---- Channel indices
OCC, SPEC, Q = 0, 1, 2
BXP, BXM, BYP, BYM, BZP, BZM = 3, 4, 5, 6, 7, 8
PHI = 9

# ---- Species → max valence (demo values; extend as needed)
MAX_SPEC_ID = 32
_max_valence_table = jnp.array(
    [0, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0] + [4] * (MAX_SPEC_ID - 11),
    dtype=jnp.uint8,
)

def lookup_max_valence(spec_id: jnp.ndarray) -> jnp.ndarray:
    idx = jnp.clip(spec_id, 0, _max_valence_table.shape[0]-1)
    return _max_valence_table[idx]

# ---- Keep channel dim helper: returns [B,1,X,Y,Z]
def ch(grid: jnp.ndarray, idx: int) -> jnp.ndarray:
    return grid[:, idx:idx+1, ...]  # slice preserves the C dimension

# ---- Bonds per voxel (sum of 6 bond channels) → [B,X,Y,Z]
def current_bonds(grid: jnp.ndarray) -> jnp.ndarray:
    bonds6 = grid[:, BXP:BZM+1, ...].astype(jnp.uint8)        # [B,6,X,Y,Z]
    return jnp.sum(bonds6, axis=1)                             # [B,X,Y,Z]

# ---- Free valence = max_valence(spec) - current_bonds (clamped 0..6) → [B,X,Y,Z]
def free_valence(grid: jnp.ndarray) -> jnp.ndarray:
    spec = grid[:, SPEC, ...]                                  # [B,X,Y,Z]
    maxv = lookup_max_valence(spec).astype(jnp.int16)
    cur  = current_bonds(grid).astype(jnp.int16)
    free = jnp.clip(maxv - cur, 0, 6).astype(jnp.uint8)
    return free

# ---- Deterministic coordinate hash (overflow-safe via uint32 math)
def coord_hash(X: int, Y: int, Z: int) -> jnp.ndarray:
    x = jnp.arange(X, dtype=jnp.uint32)[None, None, :, None, None]
    y = jnp.arange(Y, dtype=jnp.uint32)[None, None, None, :, None]
    z = jnp.arange(Z, dtype=jnp.uint32)[None, None, None, None, :]
    C1 = jnp.uint32(0x9E3779B1)
    C2 = jnp.uint32(0x7F4A7C15)
    C3 = jnp.uint32(0x94D049BB)
    h = (x * C1) ^ (y * C2) ^ (z * C3)
    h = h & jnp.uint32(0x7FFFFFFF)
    return h.astype(jnp.int32)   # [1,1,X,Y,Z]

# ---- Propose bond additions in 6 directions → [B,6,X,Y,Z] (uint8)
def propose_bonds(grid: jnp.ndarray) -> jnp.ndarray:
    _, _, X, Y, Z = grid.shape

    occ5 = ch(grid, OCC).astype(jnp.uint8)      # [B,1,X,Y,Z]
    phi5 = ch(grid, PHI).astype(jnp.int16)      # [B,1,X,Y,Z]
    free5 = free_valence(grid)[:, None, ...].astype(jnp.int16)  # [B,1,X,Y,Z]

    # neighbor views (roll on axes X=2, Y=3, Z=4)
    occ_xp = jnp.roll(occ5, +1, axis=2); occ_xm = jnp.roll(occ5, -1, axis=2)
    occ_yp = jnp.roll(occ5, +1, axis=3); occ_ym = jnp.roll(occ5, -1, axis=3)
    occ_zp = jnp.roll(occ5, +1, axis=4); occ_zm = jnp.roll(occ5, -1, axis=4)

    free_xp = jnp.roll(free5, +1, axis=2); free_xm = jnp.roll(free5, -1, axis=2)
    free_yp = jnp.roll(free5, +1, axis=3); free_ym = jnp.roll(free5, -1, axis=3)
    free_zp = jnp.roll(free5, +1, axis=4); free_zm = jnp.roll(free5, -1, axis=4)

    # existing bonds as [B,1,X,Y,Z] for uniform broadcasting
    bxp = ch(grid, BXP); bxm = ch(grid, BXM)
    byp = ch(grid, BYP); bym = ch(grid, BYM)
    bzp = ch(grid, BZP); bzm = ch(grid, BZM)

    # simple favorability: allow add if PHI does not increase
    phi_xp = jnp.roll(phi5, +1, axis=2); phi_xm = jnp.roll(phi5, -1, axis=2)
    phi_yp = jnp.roll(phi5, +1, axis=3); phi_ym = jnp.roll(phi5, -1, axis=3)
    phi_zp = jnp.roll(phi5, +1, axis=4); phi_zm = jnp.roll(phi5, -1, axis=4)

    favor_xp = (phi5 >= phi_xp).astype(jnp.uint8)
    favor_xm = (phi5 >= phi_xm).astype(jnp.uint8)
    favor_yp = (phi5 >= phi_yp).astype(jnp.uint8)
    favor_ym = (phi5 >= phi_ym).astype(jnp.uint8)
    favor_zp = (phi5 >= phi_zp).astype(jnp.uint8)
    favor_zm = (phi5 >= phi_zm).astype(jnp.uint8)

    # propose when both occupied, no existing bond, both sides have free valence, and favorable PHI
    can_add_xp = (occ5 & occ_xp) & (1 - bxp) & (free5 > 0) & (free_xp > 0) & favor_xp
    can_add_xm = (occ5 & occ_xm) & (1 - bxm) & (free5 > 0) & (free_xm > 0) & favor_xm
    can_add_yp = (occ5 & occ_yp) & (1 - byp) & (free5 > 0) & (free_yp > 0) & favor_yp
    can_add_ym = (occ5 & occ_ym) & (1 - bym) & (free5 > 0) & (free_ym > 0) & favor_ym
    can_add_zp = (occ5 & occ_zp) & (1 - bzp) & (free5 > 0) & (free_zp > 0) & favor_zp
    can_add_zm = (occ5 & occ_zm) & (1 - bzm) & (free5 > 0) & (free_zm > 0) & favor_zm

    # deterministic ownership of each edge to avoid double-add
    h    = coord_hash(X, Y, Z)                  # [1,1,X,Y,Z]
    h_xp = jnp.roll(h, +1, axis=2); h_xm = jnp.roll(h, -1, axis=2)
    h_yp = jnp.roll(h, +1, axis=3); h_ym = jnp.roll(h, -1, axis=3)
    h_zp = jnp.roll(h, +1, axis=4); h_zm = jnp.roll(h, -1, axis=4)

    win_xp = (h < h_xp).astype(jnp.uint8)
    win_xm = (h < h_xm).astype(jnp.uint8)
    win_yp = (h < h_yp).astype(jnp.uint8)
    win_ym = (h < h_ym).astype(jnp.uint8)
    win_zp = (h < h_zp).astype(jnp.uint8)
    win_zm = (h < h_zm).astype(jnp.uint8)

    prop_xp = can_add_xp & win_xp
    prop_xm = can_add_xm & win_xm
    prop_yp = can_add_yp & win_yp
    prop_ym = can_add_ym & win_ym
    prop_zp = can_add_zp & win_zp
    prop_zm = can_add_zm & win_zm

    # concatenate along channel axis → [B,6,X,Y,Z]
    props = jnp.concatenate([prop_xp, prop_xm, prop_yp, prop_ym, prop_zp, prop_zm], axis=1)
    return props.astype(jnp.uint8)

# ---- (stub) propose bond breaks (e.g., for strain); returns zeros for now
def propose_breaks(grid: jnp.ndarray) -> jnp.ndarray:
    return jnp.zeros_like(grid[:, BXP:BZM+1, ...], dtype=jnp.uint8)

# ---- Apply adds/breaks, mirroring to neighbor channels (keep 5D during ops)
def apply_bond_updates(grid: jnp.ndarray,
                       adds: jnp.ndarray,
                       breaks: jnp.ndarray) -> jnp.ndarray:
    # adds/breaks: [B,6,X,Y,Z] for +X,-X,+Y,-Y,+Z,-Z
    b5 = grid[:, BXP:BZM+1, ...].astype(jnp.uint8)   # [B,6,X,Y,Z]

    # keep channel dim to stay 5D for rolls
    add_bxp = adds[:, 0:1, ...]
    add_bxm = adds[:, 1:2, ...]
    add_byp = adds[:, 2:3, ...]
    add_bym = adds[:, 3:4, ...]
    add_bzp = adds[:, 4:5, ...]
    add_bzm = adds[:, 5:6, ...]
    brk_bxp = breaks[:, 0:1, ...]
    brk_bxm = breaks[:, 1:2, ...]
    brk_byp = breaks[:, 2:3, ...]
    brk_bym = breaks[:, 3:4, ...]
    brk_bzp = breaks[:, 4:5, ...]
    brk_bzm = breaks[:, 5:6, ...]

    # Mirror proposals: neighbor's opposite direction maps to this voxel's channel
    mirror_bxm = jnp.roll(add_bxm, +1, axis=2)  # neighbor -X rolled +X becomes our +X
    mirror_bxp = jnp.roll(add_bxp, -1, axis=2)  # neighbor +X rolled -X becomes our -X
    mirror_bym = jnp.roll(add_bym, +1, axis=3)
    mirror_byp = jnp.roll(add_byp, -1, axis=3)
    mirror_bzm = jnp.roll(add_bzm, +1, axis=4)
    mirror_bzp = jnp.roll(add_bzp, -1, axis=4)

    new_bxp = (b5[:, 0:1] | add_bxp | mirror_bxm) & (1 - brk_bxp)
    new_bxm = (b5[:, 1:2] | add_bxm | mirror_bxp) & (1 - brk_bxm)
    new_byp = (b5[:, 2:3] | add_byp | mirror_bym) & (1 - brk_byp)
    new_bym = (b5[:, 3:4] | add_bym | mirror_byp) & (1 - brk_bym)
    new_bzp = (b5[:, 4:5] | add_bzp | mirror_bzm) & (1 - brk_bzp)
    new_bzm = (b5[:, 5:6] | add_bzm | mirror_bzp) & (1 - brk_bzm)

    new_bonds = jnp.concatenate([new_bxp, new_bxm, new_byp, new_bym, new_bzp, new_bzm], axis=1).astype(jnp.uint8)
    return grid.at[:, BXP:BZM+1, ...].set(new_bonds)

# ---- Simple 6-neighbor diffusion for PHI
def diffuse_phi(grid: jnp.ndarray, diffusion_rate: int = 1) -> jnp.ndarray:
    phi5 = ch(grid, PHI).astype(jnp.int16)   # [B,1,X,Y,Z]
    phi_sum = (
        jnp.roll(phi5, +1, 2) + jnp.roll(phi5, -1, 2) +
        jnp.roll(phi5, +1, 3) + jnp.roll(phi5, -1, 3) +
        jnp.roll(phi5, +1, 4) + jnp.roll(phi5, -1, 4)
    )
    phi_avg = (phi_sum // 6)
    new_phi = jnp.clip(phi5 + diffusion_rate * (phi_avg - phi5), -32768, 32767).astype(jnp.int16)
    return grid.at[:, PHI, ...].set(new_phi[:, 0, ...])

# ---- One step; jit-compiled (TPU/GPU/CPU)
def ca_step(grid: jnp.ndarray) -> jnp.ndarray:
    adds   = propose_bonds(grid)
    breaks = propose_breaks(grid)
    grid2  = apply_bond_updates(grid, adds, breaks)
    grid3  = diffuse_phi(grid2, diffusion_rate=1)
    return grid3

ca_step_jit = jax.jit(ca_step)

# ---- Utilities for set-up
def empty_grid(batch: int, X: int, Y: int, Z: int) -> jnp.ndarray:
    return jnp.zeros((batch, 10, X, Y, Z), dtype=jnp.int32)

def place_atom(grid: jnp.ndarray, b: int, x: int, y: int, z: int,
               species: int, charge: int = 0, phi: int = 0) -> jnp.ndarray:
    grid = grid.at[b, OCC,  x, y, z].set(1)
    grid = grid.at[b, SPEC, x, y, z].set(int(species))
    grid = grid.at[b, Q,    x, y, z].set(int(charge))
    grid = grid.at[b, PHI,  x, y, z].set(int(phi))
    return grid

# ---- Demo main
def main():
    B, X, Y, Z = 1, 16, 16, 16
    g = empty_grid(B, X, Y, Z)

    # Place two atoms 1 cell apart in +X; give center a slightly higher PHI
    g = place_atom(g, 0, 8, 8, 8, species=6, phi=10)   # Carbon
    g = place_atom(g, 0, 9, 8, 8, species=8, phi=9)    # Oxygen

    print("Running CA steps...")
    for t in range(10):
        g = ca_step_jit(g)
        bc = int(current_bonds(g)[0, 8, 8, 8])
        print(f" step {t+1:02d} -> bonds at (8,8,8): {bc}")

    total_bonds = int(jnp.sum(current_bonds(g)))
    total_occ   = int(jnp.sum(g[:, OCC, ...]))
    print(f"Done. total_occ={total_occ}, total_bonds_sum={total_bonds}")

if __name__ == "__main__":
    main()
