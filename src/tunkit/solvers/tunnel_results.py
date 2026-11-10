"""
Tunneling results container

Solvers return the subclasses instead of printing. So there is a better
handling downstream and to possible manipulations later for the GUI.

A clipped or reversed path may potentially be a different effective barrier
so it is dependent on the ReactionPath feeded.

Printing
------------
report()   full output.
summary()  lean output with the most important information.

Subclasses override the basic class with their custom reports and fields.
"""

from dataclasses import dataclass, field, fields, is_dataclass
import numpy as np

from tunkit.core import constants as const








# Dataclass overrides the repr so it must be disabled.
@dataclass(repr=False, eq=False)
class TunnelingResult:
    """
    Base for the tunneling output.

    Subclasses override report() and summary().
    
    Subclasses must also use @dataclass(repr=False, eq=False). 
    """

    # Every field of the subclasses must have a default
    # The error is a confusing TypeError...
    method:    str = ''
    exec_logs: str = '' # logs acumulated during the execution of the program
    
    path:   object = None                  # the ReactionPath that produced this
    params: dict = field(default_factory=dict)   # the options it was run with

    # base printing. To be overun later
    def summary(self):
        """Lean output. """
        parts = [f"{self.method}"]
        for label, val, unit in (
                ('Level splitting', self.level_splitting, 'cm^-1'),
                ('Tunneling rate',  self.tunneling_frequency, 's^-1'),
                ('Half-life',       self.half_life, 's')):
            if val is not None:
                parts.append(f"  {label + ':':<18s}{val:.3e} {unit}")
        return '\n'.join(parts)

    def report(self):
        """Full output."""
        print(self.summary())

    def __str__(self):
        return self.summary()

    def __repr__(self):
        n = self.path.n_images if self.path is not None else '?'
        return (f"{type(self).__name__}(method={self.method!r}, "
                f"nu={self.tunneling_frequency}, images={n})")


    def plot(self, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} does not define a plot yet")








@dataclass
class VibAnalysis:
    """
    Output of a single frequency analysis.

    The class is separated because calc_tunneling_wkb_asym runs the 
    analysis on forward and reverse paths and has to keep both.
    """

    label:         str        = ''     # 'Reactant' / 'Product'
    winner_freq:   float      = None   # winning mode, cm^-1
    freqs:         np.array   = None   # each frequency
    overlaps:      np.array   = None   # the amount of overlap between the tangent and each mode
    contributions: np.array   = None   # the normalized contribution of the vibration to the first step
    mode_table:    np.ndarray = None   # (n_modes, 4): index, freq, overlap, contribution %
    hessian_idx:   int        = None
    gradient_idx:  int        = None
    rms_force:     float      = None
    failed:        bool       = False   # analysis could not be completed

    def _text(self):
        if self.failed or self.mode_table is None:
            return (f"{'-' * 48}\n"
                    f"Attack Frequency Analysis - {self.label}\n"
                    f"{'-' * 48}\n"
                    f"Analysis failed. Check the curve and the frequencies.\n"
                    f"{'-' * 48}")

        rows = '\n'.join(
            f"{int(i):<5} | {f:<12.2f} | {o:<10.4f} | {c:<5.1f}%"
            for i, f, o, c in self.mode_table)

        return f"""\
{'-' * 48}
Attack Frequency Analysis - {self.label}
{'-' * 48}
{'Mode':<5} | {'Freq (cm-1)':<12} | {'Overlap':<10} | {'Contribution':<10}
{'-' * 48}
{rows}
{'-' * 48}
Hessian Source:        Step {self.hessian_idx} (Reactant)
Gradient Source:       Step {self.gradient_idx}
RMS Force:             {self.rms_force:.2e}
Attack Freq:           {self.winner_freq:.2f} cm^-1
{'-' * 48}"""

    def report(self):
        print(self._text())

    def __str__(self):
        return self._text()











@dataclass
class WkbAsymDwpResult(TunnelingResult):

    method: str = 'Rastelli (2012) Asymmetric WKB'

    # VibAnalysis class for the reactant and for the product
    reactant_vib: VibAnalysis = None
    product_vib:  VibAnalysis = None

    # potential
    U_L: float = None
    U_R: float = None

    # pre exponential factor
    A_eta: float = None

    # results
    coupling_cm:         float = None
    tun_amplitude:       float = None
    level_splitting:     float = None
    tunneling_frequency: float = None
    half_life:           float = None

    def summary(self):
        return (f"{self.method}\n"
                f"  Level splitting:  {self.level_splitting:.3e} cm^-1\n"
                f"  Tunneling rate:   {self.tunneling_frequency:.3e} s^-1\n"
                f"  Half-life:        {self.half_life:.3e} s")

    def report(self):
        self.reactant_vib.report()
        self.product_vib.report()
        print(f"""\
{'-' * 50}
{self.method}
{'-' * 50}
Freq L (Reactant): {self.reactant_vib.winner_freq:.2f} cm-1
Freq R (Product):  {self.product_vib.winner_freq:.2f} cm-1
Barrier Height L:  {self.U_L * const.hartree_to_kcalmol:.2f} kcal/mol
Barrier Height R:  {self.U_R * const.hartree_to_kcalmol:.2f} kcal/mol
Asymmetry Factor:  {self.A_eta:.4f}
Coupling Matrix Element (V): {self.coupling_cm:.3e} cm^-1
Level splitting: {self.level_splitting:.3e} cm^-1
Tunneling rate: {self.tunneling_frequency:.3e} s^-1
Half-Life: {self.half_life:.3e} s
{'-' * 50}""")








@dataclass(repr=False, eq=False)
class WKBSymDWPResult(TunnelingResult):
    """Semiclassical WKB tunneling through a symmetric barrier."""

    method: str = 'WKB semiclassical symmetric DWP'

    # VibAnalysis class for the vibrational analysis and posterior printing
    vib: VibAnalysis = None                # the mode analysis and its table

    # the results
    tunneling_frequency: float = None      # s^-1
    half_life:           float = None      # s
    level_splitting:     float = None      # cm^-1

    # action integral parameters
    action_integral:       float = None    # a.u.
    action_integral_error: float = None    # quad's error estimate
    theta:                 float = None    # WKB exponent
    permeability:          float = None    # dwp=False branch only
    pre_exponential:           float = None    # dwp=True branch only, exp(-theta)

    roots:         np.ndarray = None       # the two turning points, m
    turning_point: float      = None       # tunneling energy, Eh

    # curves for plotting later
    V_minus_E:    np.ndarray = None
    positive_rp:  np.ndarray = None        # sampling between the roots
    sqroot:       np.ndarray = None        # sqrt(V - E) on positive_rp
    wkb_energies: np.ndarray = None

    # parameters
    kinetic_rate: bool = None              # irreversible decay, doubles the rate
    reduced_mass: float = None             # amu, as supplied
    raw_energies: bool = None              # unprojected energies were used

    def summary(self):
        parts = [f"{self.method}"]
        if self.relevant_vib is not None:
            parts.append(f"  Attack frequency:     {self.relevant_vib:.2f} cm^-1")
        if self.theta is not None:
            parts.append(f"  WKB exponent (theta): {self.theta:.4f}")
        parts.append(f"  Tunneling frequency:  {self.tunneling_frequency:.3e} s^-1")
        if self.level_splitting is not None:
            parts.append(f"  Level splitting:      {self.level_splitting:.3e} cm^-1")
        if self.half_life is not None:
            parts.append(f"  Half-life:            {self.half_life:.3e} s")
        return '\n'.join(parts)

    def report(self, modes=True):
        """
        Full report. 

        modes: False skips the per-mode overlap table printing only
                printing only the tunneling results.

        """
        if modes and self.vib is not None:
            self.vib.report()

        if self.raw_energies:
            print("WARNING! Nonprojected energies were used, mind you...")

        if self.turning_point is not None:
            print(f"Tunneling energy (turning point): {self.turning_point:.6f} Eh")
        if self.roots is not None:
            print(f"Integration limits: {self.roots[0]:.2e} to "
                  f"{self.roots[1]:.2e} m")

        err = ('' if self.action_integral_error is None
               else f" (Error est: {self.action_integral_error:.2e})")
        print(f"Action Integral: {self.action_integral:.4e}{err}")
        print(f"WKB Exponent (Theta) = {self.theta:.4f}")

        if self.permeability is not None:
            print(f"Barrier Permeability = {self.permeability:.4e}")
        if self.exponential is not None:
            print(f"exp(-Theta) = {self.exponential:.4e}")

        print(f"Tunneling frequency is {self.tunneling_frequency:.3e} s-1")
        if self.level_splitting is not None:
            print(f"Energy level splitting is {self.level_splitting:.3e} cm-1")
        if self.half_life is not None:
            print(f"Half-life: {self.half_life:.3e} s")

        formula = ('nu = (nu_0/pi) * exp(-Theta)')
        note = ', with doubled Theta for irreversible decay' if self.kinetic_rate else ''
        print(f"({formula}{note}; reduced mass {self.reduced_mass} amu)")

    # plotting

    # def plot(self, show=True, **kwargs):
    #     """Draw the barrier integrand between the turning points."""
    #     from ..plotting.results import plot_action_integral
    #     return plot_action_integral(self, show=show, **kwargs)








@dataclass(repr=False, eq=False)
class WKBResult(TunnelingResult):
    """Semiclassical WKB tunneling through the barrier."""

    method: str = 'WKB semiclassical transmission probability.' 

    # VibAnalysis class for the vibrational analysis and posterior printing
    vib: VibAnalysis = None                # the mode analysis and its table

    # action integral parameters
    action_integral:       float = None # a.u.
    action_integral_error: float = None # quad's error estimate
    theta:                 float = None # WKB exponent
    permeability:          float = None # dwp=False branch only
    pre_exponential:       float = None # dwp=True branch only, exp(-theta)

    roots:         np.ndarray = None # the two turning points, m
    turning_point: float      = None # tunneling energy, Eh

    # curves for plotting later
    V_minus_E:    np.ndarray = None
    positive_rp:  np.ndarray = None # sampling between the roots
    sqroot:       np.ndarray = None # sqrt(V - E) on positive_rp
    wkb_energies: np.ndarray = None # It's just the projected energies, but it's easier to access this way

    # parameters
    reduced_mass: float = None # amu, as supplied
    raw_energies: bool  = None # unprojected energies were used


    # the results
    tunneling_frequency: float = None # s^-1
    half_life:           float = None # s

    def summary(self):
        parts = [f"{self.method}"]
        if self.vib is not None:
            parts.append(f"  Attack frequency:     {self.vib.winner_freq:.2f} cm^-1")
        if self.theta is not None:
            parts.append(f"  WKB exponent (theta): {self.theta:.4f}")
        parts.append(f"  Tunneling frequency:  {self.tunneling_frequency:.3e} s^-1")
        if self.level_splitting is not None:
            parts.append(f"  Level splitting:      {self.level_splitting:.3e} cm^-1")
        if self.half_life is not None:
            parts.append(f"  Half-life:            {self.half_life:.3e} s")
        return '\n'.join(parts)

    def report(self, modes=True):
        """
        Full report. 

        modes: False skips the per-mode overlap table printing only
                printing only the tunneling results.

        """
        if modes and self.vib is not None:
            self.vib.report()

        if self.raw_energies:
            print("WARNING! Nonprojected energies were used, mind you...")

        if self.turning_point is not None:
            print(f"Tunneling energy (turning point): {self.turning_point:.6f} Eh")
        if self.roots is not None:
            print(f"Integration limits: {self.roots[0]:.2e} to "
                  f"{self.roots[1]:.2e} m")

        err = ('' if self.action_integral_error is None
               else f" (Error est: {self.action_integral_error:.2e})")
        print(f"Action Integral: {self.action_integral:.4e}{err}")
        print(f"WKB Exponent (Theta) = {self.theta:.4f}")

        if self.permeability is not None:
            print(f"Barrier Permeability = {self.permeability:.4e}")
        if self.exponential is not None:
            print(f"exp(-Theta) = {self.exponential:.4e}")

        print(f"Tunneling frequency is {self.tunneling_frequency:.3e} s-1")
        if self.level_splitting is not None:
            print(f"Energy level splitting is {self.level_splitting:.3e} cm-1")
        if self.half_life is not None:
            print(f"Half-life: {self.half_life:.3e} s")

        formula = ('nu = (nu_0/pi) * exp(-Theta)') 
        print(f"({formula}; reduced mass {self.reduced_mass} amu)")









@dataclass(repr=False, eq=False)
class AnalyticResult(TunnelingResult):
    """Common ground for the two analytic double-well solvers."""

    # levels, already converted to avoid conversion later
    energy_levels_J:        np.ndarray = None
    energy_levels_cm:       np.ndarray = None
    energy_levels_kcalmol:  np.ndarray = None
    leakage_percent:        np.ndarray = None
    theta_deg:              np.ndarray = None
    coherences:             np.ndarray = None

    # solved levels, full precision 
    energy_levels: np.ndarray = None      # hartree, final iteration

    # fitted potential parameters, full convergence histories
    h_vec: np.ndarray = None
    b_vec: np.ndarray = None
    L_vec: np.ndarray = None

    # splitting, already converted
    splitting_hartree: float = None
    splitting_J:       float = None
    splitting_cm:      float = None

    tunneling_frequency: float = None

    kinetic_rate: bool = None

    # shared rendering
    def _level_table_text(self, width=90):
        """Helper to build the level table"""
        if self.level_energies_J is None:
            return "WARNING! No computed levels found!"

        table = ''
        table += '=' * width + '\n'

        table += (f"{'Level':<5} | {'Energy (kcal/mol)':<17} | {'Energy (cm^-1)':<15} | "
                  f"{'Leakage (%)':<16} | {'Theta (deg)':<10}\n")

        table += '-' * width + '\n'

        for i in range(len(self.level_energies_J)):
            table += f"{i:<5} | {float(self.level_energies_kcalmol[i]):>15.4e} " +\
                  f"| {float(self.level_energies_cm[i]):>15.2f} " +\
                  f"| {float(self.leakage_percent[i]):>16.4f} " +\
                  f"| {float(self.theta_deg[i]):>10.4f}\n"

        table += '=' * width + '\n'

        return table

    def summary(self):
        return (f"{self.method}\n"
                f"  Ground state splitting:  {self.splitting_cm:.4e} cm^-1\n"
                f"  Associated frequency:    {self.tunneling_frequency:.4e} s^-1"
                + (f"\n  Half-life:               {self.half_life:.4e} s"
                   if self.half_life is not None else ''))








@dataclass(repr=False, eq=False)
class AnalyticSymResult(AnalyticResult):
    """Symmetric double well, solved on the oblate spheroidal basis."""

    method: str = 'Analytic symmetric double well (APS)'

    m_vec: np.ndarray = None              # m vector
    aps_energy_levels: np.ndarray = None  # dimensionless eigenvalues

    # E
    well_depth_hartree: float = None

    def report(self, width=90):
        print(self._level_table_text(width))

        if self.m_vec is not None:
            print("Parameters final values:")
            print(f"m = {int(self.m_vec[-1]):d}, h = {self.h_vec[-1]:.4e}, "
                  f"b = {self.b_vec[-1]:.3e}, L = {self.L_vec[-1]:.5e}")

        print(f"Ground state splitting (J): {self.splitting_J:.4e} J")
        print(f"Ground state splitting (cm-1): {self.splitting_cm:.4e} cm-1")
        print(f"Associated Frequency:   {self.tunneling_frequency:.4e} s^-1")

        # if self.well_depth_hartree is not None:
        #     print(f"(levels referenced to the well bottom, "
        #           f"offset {self.well_depth_hartree:.4e} Eh)")








@dataclass(repr=False, eq=False)
class AnalyticAsymResult(AnalyticResult):
    """Asymmetric double well, solved by finite differences."""

    method: str = 'Analytic asymmetric double well'

    a_vec: np.ndarray = None              # asymmetry parameter history
    che_energy_levels: np.ndarray = None

    def report(self, width=90):
        print(self._level_table_text(width))

        if self.h_vec is not None:
            print("Parameters final values:")
            print(f"h = {self.h_vec[-1]:.4e}, b = {self.b_vec[-1]:.4e}, "
                  f"a = {self.a_vec[-1]:.4e}, L = {self.L_vec[-1]:.5e}")

        print(f"Ground state splitting: {self.splitting_J:.4e} J")
        print(f"Ground state splitting: {self.splitting_cm:.4e} cm-1")
        print(f"Associated Frequency:   {self.tunneling_frequency:.4e} s^-1")

        if self.a_vec is not None and self.a_vec[-1] != 0:
            print("Note: In asymmetric wells, splitting includes the "
                  "asymmetry offset (detuning).")








@dataclass(repr=False, eq=False)
class MatrixResult(AnalyticAsymResult):
    """Asymmetric double-well analysis plus an irreversible matrix rate."""

    method: str = 'Matrix tunneling (irreversible)'

    # conditions 
    temperature:       float = None        # K
    reactant_well:     str = None          # 'left' / 'right'
    starting_well:     str = None          # defined by the previous solver
    theta_threshold:   float = None        # defined by the previous solver
    harm_attempt_freq: bool = None         #
    omega_eff_s:       float = None        # effective attempt frequency, s^-1
    n_channels:        int = None

    # channel parameters (good for debug and getting more information)
    channel_index:   np.ndarray = None     # product state index
    t_ij_au:         np.ndarray = None     # coupling matrix element, a.u.
    rate_ij_s:       np.ndarray = None     # channel rate, s^-1
    delta_E_au:      np.ndarray = None     # product minus reactant energy, a.u.
    thermal_penalty: np.ndarray = None     # Boltzmann factor, 1.0 when downhill

    # most important results
    total_rate_s: float = None             # s^-1, irreversible
    half_life_s:  float = None             # s

    @classmethod
    def from_analytic(cls, analytic, **current_fields):
        """
        Helper to populate the current matrix results from the previous 
        (and required) asymmetric tunneling computation. Every field is 
        copied automatically, even future updated fields.

        """
        carried = {f.name: getattr(analytic, f.name)
                   for f in fields(AnalyticAsymResult)
                   if f.name != 'method'}
        carried.update(current_fields)
        return cls(**carried)



    def summary(self):
        dominant_channel = int(self.channel_index[int(np.argmax(self.rate_ij_s))])

        parts = [f"\n--- Matrix Tunneling Analysis ---"]
        if self.temperature is not None:
            parts.append(f"  Temperature:          {self.temperature:.1f} K")
        if self.total_rate_s is not None:
            parts.append(f"  Total tunneling rate: {self.total_rate_s:.4e} s^-1")
            parts.append(f"  Half-life:            {self.half_life_s:.4e} s")
        if self.n_channels:
            parts.append(f"  Channels:             {self.n_channels} "
                         f"(dominant: 0 -> {dominant_channel})")
        return '\n'.join(parts)



    def report(self, width=90, levels=True):
        """
        Full report: the asymmetric analysis via matrix.

        """
        if levels:
            print(self._level_table_text(width))
            if self.h_vec is not None:
                print("Parameters final values:")
                print(f"h = {self.h_vec[-1]:.4e}, b = {self.b_vec[-1]:.4e}, "
                      f"a = {self.a_vec[-1]:.4e}, L = {self.L_vec[-1]:.5e}")
            if self.splitting_cm is not None:
                print(f"Ground state splitting: {self.splitting_cm:.4e} cm-1")
                print("Note: In asymmetric wells, splitting includes the asymmetry offset (detuning).")

        rule = '-' * 50
        print(f"\n--- Matrix Tunneling Analysis (T={self.temperature}K) ---")
        print(f"Reactant well: {str(self.reactant_well).capitalize()}")
        print(f"Effective Attempt Frequency: {self.omega_eff_s:.4e} s^-1")
        print(rule)

        if self.n_channels:
            for p, t, r in zip(self.channel_index, self.t_ij_au, self.rate_ij_s):
                print(f"Channel 0 -> {int(p)}: t = {float(t):.4e} a.u. | "
                      f"Rate = {float(r):.4e} s^-1")
        else:
            print("No channels passed the theta threshold "
                  f"({self.theta_threshold} deg).")

        print(rule)
        print(f"Total tunneling rate: {self.total_rate_s:.4e} s^-1")
        print(f"Half-life: {self.half_life_s:.4e} s")

        if self.thermal_penalty is not None and np.any(self.thermal_penalty < 1.0):
            n_uphill = int(np.sum(self.thermal_penalty < 1.0))
            print(f"Note: {n_uphill} channel(s) are uphill and carry a "
                  f"Boltzmann penalty at {self.temperature} K.")












