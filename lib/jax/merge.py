# --- Channel-safe single-channel views (keep C dim)
def ch(grid, idx):
    return grid[:, idx:idx+1, ...]  # shape [B,1,X,Y,Z]

# ---- Count current bonds per cell (sum of 6 bond bits)
def current_bonds(grid):
    bonds6 = grid[:, BXP:BZM+1, ...].astype(jnp.uint8)   # [B,6,X,Y,Z]
    return jnp.sum(bonds6, axis=1)                       # [B,X,Y,Z]

def free_valence(grid):
    spec = grid[:, SPEC, ...]                            # [B,X,Y,Z] ok for lookup
    maxv = lookup_max_valence(spec)
    cur  = current_bonds(grid)
    free = jnp.clip(maxv.astype(jnp.int16) - cur.astype(jnp.int16), 0, 6)
    return free.astype(jnp.uint8)                        # [B,X,Y,Z]

# ---- Propose bond additions (uses 5D tensors with C=1 for safe rolling)
def propose_bonds(grid):
    B, C, X, Y, Z = grid.shape

    occ5 = ch(grid, OCC).astype(jnp.uint8)              # [B,1,X,Y,Z]
    phi5 = ch(grid, PHI).astype(jnp.int16)              # [B,1,X,Y,Z]
    free4 = free_valence(grid).astype(jnp.int16)        # [B,X,Y,Z]
    free5 = free4[:, None, ...]                         # [B,1,X,Y,Z]

    # neighbor views via rolls on axes 2,3,4
    occ_xp = jnp.roll(occ5, +1, axis=2); occ_xm = jnp.roll(occ5, -1, axis=2)
    occ_yp = jnp.roll(occ5, +1, axis=3); occ_ym = jnp.roll(occ5, -1, axis=3)
    occ_zp = jnp.roll(occ5, +1, axis=4); occ_zm = jnp.roll(occ5, -1, axis=4)

    free_xp = jnp.roll(free5, +1, axis=2); free_xm = jnp.roll(free5, -1, axis=2)
    free_yp = jnp.roll(free5, +1, axis=3); free_ym = jnp.roll(free5, -1, axis=3)
    free_zp = jnp.roll(free5, +1, axis=4); free_zm = jnp.roll(free5, -1, axis=4)

    # existing bonds as [B,1,X,Y,Z] for uniform math
    bxp = ch(grid, BXP); bxm = ch(grid, BXM)
    byp = ch(grid, BYP); bym = ch(grid, BYM)
    bzp = ch(grid, BZP); bzm = ch(grid, BZM)

    # favor bonds when PHI does not increase
    phi_xp = jnp.roll(phi5, +1, axis=2); phi_xm = jnp.roll(phi5, -1, axis=2)
    phi_yp = jnp.roll(phi5, +1, axis=3); phi_ym = jnp.roll(phi5, -1, axis=3)
    phi_zp = jnp.roll(phi5, +1, axis=4); phi_zm = jnp.roll(phi5, -1, axis=4)

    favor_xp = (phi5 >= phi_xp).astype(jnp.uint8)
    favor_xm = (phi5 >= phi_xm).astype(jnp.uint8)
    favor_yp = (phi5 >= phi_yp).astype(jnp.uint8)
    favor_ym = (phi5 >= phi_ym).astype(jnp.uint8)
    favor_zp = (phi5 >= phi_zp).astype(jnp.uint8)
    favor_zm = (phi5 >= phi_zm).astype(jnp.uint8)

    # Propose where both occupied, no existing bond, free valence on both, favorable PHI
    can_add_xp = (occ5 & occ_xp) & (1 - bxp) & (free5 > 0) & (free_xp > 0) & favor_xp
    can_add_xm = (occ5 & occ_xm) & (1 - bxm) & (free5 > 0) & (free_xm > 0) & favor_xm
    can_add_yp = (occ5 & occ_yp) & (1 - byp) & (free5 > 0) & (free_yp > 0) & favor_yp
    can_add_ym = (occ5 & occ_ym) & (1 - bym) & (free5 > 0) & (free_ym > 0) & favor_ym
    can_add_zp = (occ5 & occ_zp) & (1 - bzp) & (free5 > 0) & (free_zp > 0) & favor_zp
    can_add_zm = (occ5 & occ_zm) & (1 - bzm) & (free5 > 0) & (free_zm > 0) & favor_zm

    # Deterministic owner for each edge via coordinate hash
    x = jnp.arange(X)[None, None, :, None, None]
    y = jnp.arange(Y)[None, None, None, :, None]
    z = jnp.arange(Z)[None, None, None, None, :]

    def voxel_hash(x, y, z):
        h = (x * 0x9E3779B1) ^ (y * 0x7F4A7C15) ^ (z * 0x94D049BB)
        return (h & 0x7FFFFFFF).astype(jnp.int32)

    h    = voxel_hash(x, y, z)
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

    # Concatenate along channel axis → [B,6,X,Y,Z]
    props = jnp.concatenate([prop_xp, prop_xm, prop_yp, prop_ym, prop_zp, prop_zm], axis=1)
    return props.astype(jnp.uint8)

# ---- Simple PHI diffusion (keep channel dim)
def diffuse_phi(grid, diffusion_rate=1):
    phi5 = ch(grid, PHI).astype(jnp.int16)    # [B,1,X,Y,Z]
    phi_sum = (
        jnp.roll(phi5, +1, 2) + jnp.roll(phi5, -1, 2) +
        jnp.roll(phi5, +1, 3) + jnp.roll(phi5, -1, 3) +
        jnp.roll(phi5, +1, 4) + jnp.roll(phi5, -1, 4)
    )
    phi_avg = (phi_sum // 6)
    new_phi = phi5 + diffusion_rate * (phi_avg - phi5)
    new_phi = jnp.clip(new_phi, -32768, 32767).astype(jnp.int16)
    return grid.at[:, PHI, ...].set(new_phi[:, 0, ...])  # write back without the extra C
