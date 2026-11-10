import math
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
from scipy.optimize import brentq
from scipy.integrate import quad








def print_and_save(*args, warnings_out='', separator=' ', verbose=False):

    msg = separator.join(str(arg) for arg in args)
    if verbose:
        print(msg)

    return warnings_out + '\n' + msg









def calc_half_life(tunneling_frequency, kinetic_rate=None, unit='s', logger=None):
    if kinetic_rate:
        # Irreversible decay: t_1/2 = ln(2) / k
        # tunneling_frequency was already doubled when kinetic_rate=True
        half_life = math.log(2) / tunneling_frequency
    else:
        # Coherent oscillation: time to first transfer = 1 / (2 * nu)
        half_life = 1.0 / (2 * tunneling_frequency)
    half_life = float(half_life)
    if unit == 's':
        if half_life < 0.001:
            logger.log('The half life is {:.3e} s'.format(half_life))
        else:
            logger.log('The half life is {:.3f} s'.format(half_life))
    if unit == 'm':
        if half_life < 0.001:
            logger.log('The half life is {:.3e} m'.format(half_life / 60))
        else:
            logger.log('The half life is {:.3f} m'.format(half_life / 60))
    if unit == 'h':
        if half_life < 0.001:
            logger.log('The half life is {:.3e} h'.format(half_life / 60 / 60))
        else:
            logger.log('The half life is {:.3f} h'.format(half_life / 60 / 60))
    if unit == 'd':
        if half_life < 0.001:
            logger.log('The half life is {:.3e} days'.format(half_life / 60 / 60 / 24))
        else:
            logger.log('The half life is {:.3f} days'.format(half_life / 60 / 60 / 24))
    if unit == 'y':
        if half_life < 0.001:
            logger.log('The half life is {:.3e} years'.format(half_life / 60 / 60 / 24 / 365))
        else:
            logger.log('The half life is {:.3e} years'.format(half_life / 60 / 60 / 24 / 365))

    return half_life








def calculate_classical_frequency(x_array, V_array, E_target, mass, min_idx, barrier_idx, reactant_well):
    """
    Calculates the exact classical attempt frequency (omega) for a particle
    in an arbitrary anharmonic potential well at a specific energy level.
    
    Parameters:
    -----------
    x_array      : array of spatial coordinates (bohr)
    V_array      : array of potential energies (hartree)
    E_target     : target energy level (hartree)
    mass         : reduced mass (atomic units)
    min_idx      : array index of the well minimum
    barrier_idx  : array index of the barrier peak
    is_left_well : boolean, True if calculating for the left well
    """
    # creating the spline of the potential (should probably be a class variable?)
    V_spline = interpolate(x_array, V_array, k=3)
    
    x_min = x_array[min_idx]
    x_bar = x_array[barrier_idx]
    
    # 1.A Find classical turning points where V(x) = E_target
    try:
        if reactant_well == 'left':
            # Root 1: Inner wall (between minimum and barrier)
            x_inner = brentq(lambda x: V_spline(x) - E_target, x_min, x_bar)
            # Root 2: Outer wall (away from barrier)
            x_outer = brentq(lambda x: V_spline(x) - E_target, x_array[0], x_min)
        else:
            x_inner = brentq(lambda x: V_spline(x) - E_target, x_bar, x_min)
            x_outer = brentq(lambda x: V_spline(x) - E_target, x_min, x_array[-1])
    except ValueError:
        raise ValueError("Could not find classical turning points. Check if E_target is above the barrier or below the minimum.")
        
    x_lo, x_hi = sorted([x_inner, x_outer])
    
    # 2.A Integrate the classical period
    # T = sqrt(2m) * integral [ 1 / sqrt(E - V(x)) ] dx
    integrand = lambda x: 1.0 / np.sqrt(max(E_target - V_spline(x), 1e-30))
    period_int, _ = quad(integrand, x_lo, x_hi, limit=200)
    
    T_period_au = np.sqrt(2.0 * mass) * period_int
    omega_au = 2.0 * np.pi / T_period_au
    
    return omega_au






# class GenSolverMixin:









