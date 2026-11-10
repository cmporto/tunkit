import numpy as np







def _neb_fire_step(positions, forces, state,
                dt_start=0.1, dt_max=0.5, max_move=0.2):
    """
    FIRE optimizer (Bitzek 2006).

    Works identically for global and local - caller decides what to
    concatenate. Parameters match ORCA's FIRE defaults.

    Parameters
    ----------
    positions : ndarray (N_dof,)
    forces    : ndarray (N_dof,)
    state     : dict - mutated in place.
    dt_start  : float - initial timestep. Default 0.1.
    dt_max    : float - maximum timestep. Default 0.5.
    max_move  : float - max component cap. Default 0.2.

    Returns
    -------
    new_positions : ndarray (N_dof,)
    """

    F_INC     = 1.1
    F_DEC     = 0.5
    ALPHA_0   = 0.1
    F_ALPHA   = 0.99
    N_MIN     = 5

    if 'v' not in state:
        state['v']     = np.zeros_like(positions)
        state['dt']    = dt_start
        state['alpha'] = ALPHA_0
        state['n_pos'] = 0

    v     = state['v']
    dt    = state['dt']
    alpha = state['alpha']

    # Power
    P = float(np.dot(forces, v))

    # Mix velocity toward force direction
    v_norm = float(np.linalg.norm(v))
    f_norm = float(np.linalg.norm(forces))
    if f_norm > 1e-20:
        v = (1.0 - alpha) * v + alpha * v_norm * (forces / f_norm)

    # Adaptive logic
    if P > 0:
        state['n_pos'] += 1
        if state['n_pos'] > N_MIN:
            dt = min(dt * F_INC, dt_max)
            alpha *= F_ALPHA
    else:
        state['n_pos'] = 0
        v = np.zeros_like(v)
        dt *= F_DEC
        alpha = ALPHA_0

    # Verlet
    v = v + forces * dt
    step = v * dt

    # Trust-radius cap
    step_max = float(np.max(np.abs(step)))
    if step_max > max_move:
        # the correct way is to scale v too, so it's not saved
        # as if it were increasing while it was capped
        #step *= max_move / step_max
        scale = max_move / step_max
        step *= scale
        v *= scale
    
    state['v']     = v
    state['dt']    = dt
    state['alpha'] = alpha
    return positions + step








def _neb_lbfgs_step_per_image(position, 
                                force, 
                                state,
                                mem=20, 
                                dr=1.0e-3, 
                                max_move=0.1):
    """
    Per-image L-BFGS step with trust radius and FD initialization.

    Mirrors ORCA's NEB optimizer defaults: history depth 20, FD probe
    of size 1e-3 on the first call, MaxMove cap at 0.1 (MW units),
    history flush when MaxMove bites.

    All quantities are in mass-weighted coordinates: position in
    bohr*amu^(1/2), force in Eh/(bohr*amu^(1/2)).

    Parameters
    ----------
    position : ndarray (3*N_atoms,)
        Flattened MW position of one image.
    force : ndarray (3*N_atoms,)
        Flattened NEB force on that image (already perp + spring).
    state : dict
        Per-image optimizer state. Mutated in place. Empty dict on
        first call.
    mem : int
        L-BFGS history depth. Default 20 (matches ORCA LBFGS_Mem).
    dr : float
        FD probe step size used to initialize the curvature.
        Default 1e-3 (matches ORCA LBFGS_DR).
    max_move : float
        Trust-radius cap on the step's max component, in MW units.
        Default 0.1 (matches ORCA MaxMove).

    Returns
    -------
    new_position : ndarray (3*N_atoms,)
    """
    g = -force  # L-BFGS works on the gradient, not the force

    # If first call take an FD probe along -g
    if 'initialized' not in state or not state['initialized']:
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-14:
            # Already at a stationary point for this image
            state['initialized']  = True
            state['s_hist']       = []
            state['y_hist']       = []
            state['gamma']        = 1.0
            state['prev_q']       = position.copy()
            state['prev_g']       = g.copy()
            return position.copy()

        direction = -g / g_norm
        step = dr * direction
        state['s_hist']     = []
        state['y_hist']     = []
        state['gamma']      = None      # filled in next call!
        state['prev_q']     = position.copy()
        state['prev_g']     = g.copy()
        state['initialized'] = True
        state['fd_probe']   = True      # flag next call closes the FD pair
        return position + step

    # Build s and y from the previous step 
    s = position - state['prev_q']
    y = g - state['prev_g']
    sy = float(np.dot(s, y))

    if state.get('fd_probe', False):
        # Closing the FD probe: set the initial inverse-Hessian scale
        # gamma = (s . y) / (y . y), if curvature is positive
        yy = float(np.dot(y, y))
        if sy > 1e-12 and yy > 1e-20:
            state['gamma'] = sy / yy
        else:
            # Bad curvature on the probe so fall back to a small scale (maybe improve)
            state['gamma'] = max(dr / max(np.linalg.norm(g), 1e-3), 1e-4)
        state['fd_probe'] = False
        # Won't store this (s, y) pair as the FD step is too small to be informative for the limited-memory history 
        # Real pairs will start accumulating from the next call
    else:
        # Accept the (s, y) pair if it satisfies the curvature condition
        if sy > 1e-12:
            state['s_hist'].append(s)
            state['y_hist'].append(y)
            if len(state['s_hist']) > mem:
                state['s_hist'].pop(0)
                state['y_hist'].pop(0)
            # Update gamma from the most recent pair (Nocedal & Wright 7.20)
            yy = float(np.dot(y, y))
            if yy > 1e-20:
                state['gamma'] = sy / yy

    # Two-loop recursion: r = H * g  
    # H = inverse Hessian
    q = g.copy()
    alphas = []
    rhos = []
    for s_k, y_k in zip(reversed(state['s_hist']),
                        reversed(state['y_hist'])):
        rho_k = 1.0 / float(np.dot(y_k, s_k))
        a_k = rho_k * float(np.dot(s_k, q))
        q = q - a_k * y_k
        alphas.append(a_k)
        rhos.append(rho_k)

    gamma = state['gamma'] if state['gamma'] is not None else 1.0
    r = gamma * q

    for (s_k, y_k), a_k, rho_k in zip(
        zip(state['s_hist'], state['y_hist']),
        reversed(alphas),
        reversed(rhos),
    ):
        b_k = rho_k * float(np.dot(y_k, r))
        r = r + (a_k - b_k) * s_k

    direction = -r  # search direction

    # Trust-radius cap, from max_move with history flush on activation 
    step_max = float(np.max(np.abs(direction)))
    if step_max > max_move:
        direction = direction * (max_move / step_max)
        # Flush history and re-trigger FD probe on the next call (this has been a problem...)
        # The capped step should still be applied this iteration
        # the flush only affects the next one (ORCA Restart_On_Maxmove behavior)
        state['s_hist']      = []
        state['y_hist']      = []
        state['gamma']       = None
        state['initialized'] = False  # next call will FD-probe again

    # Update bookkeeping and return 
    state['prev_q'] = position.copy()
    state['prev_g'] = g.copy()
    return position + direction








def _neb_vpo_step(positions, forces, state,
                  dt=0.5, max_move=0.2):
    """
    Velocity Projected Optimizer (Quick-Min), Jónsson 1998.

    Works identically for global (whole band as one vector) and
    local (single image) - the caller decides what to concatenate.

    Order of operations:
    1. Velocity update:  v += F * dt
    2. Projection:       v = (v · F̂) F̂  if v·F > 0, else v = 0
    3. Step:             Δx = v * dt
    4. Trust-radius cap: scale step + velocity if max component > max_move

    Parameters
    ----------
    positions : ndarray (N_dof,)
    forces    : ndarray (N_dof,)
    state     : dict - mutated in place. Holds 'v'.
    dt        : float - Verlet timestep. Default 0.5 
    max_move  : float - max component cap. Default 0.2 

    Returns
    -------
    new_positions : ndarray (N_dof,)
    """

    if 'v' not in state:
        state['v'] = np.zeros_like(positions)

    v = state['v']
    f_norm = float(np.linalg.norm(forces))

    if f_norm < 1e-14:
        state['v'] = np.zeros_like(positions)
        return positions.copy()

    # velocity update
    v = v + forces * dt

    # Projection and sign check
    f_hat = forces / f_norm
    vf = float(np.dot(v, f_hat))
    if vf > 0.0:
        v = vf * f_hat
    else:
        v = np.zeros_like(positions)

    step = v * dt

    # 4. Max_move cap 
    step_max = float(np.max(np.abs(step)))
    if step_max > max_move:
        scale = max_move / step_max
        step *= scale
        v *= scale

    state['v'] = v
    return positions + step
