"""
MWNEB results container.

runner.run_neb returns a NebResult instead of writing onto a class, the same
way the solvers return TunnelingResult subclasses. The optimized band itself
is result.path, a fresh ReactionPath (only the molecule's invariants are
carried over from the input path, since the band has new geometries and
possibly a different number of images).

Printing
------------
report()   full output.
summary()  lean output with the most important information.
"""

from dataclasses import dataclass, field
import numpy as np
from tunkit.core import constants as const








@dataclass(repr=False, eq=False)
class NebResult:
    """Output of runner.run_neb."""

    method: str = 'Mass-weighted NEB'

    path:   object = None                       # ReactionPath with the final band
    params: dict = field(default_factory=dict)  # the options it was run with

    converged:      bool = False
    n_steps:        int = 0          # force evaluations of the movable images
    fmax_history:   list = field(default_factory=list)  # max |F_perp| per step, Eh/bohr
    frms_history:   list = field(default_factory=list)  # rms |F_perp| per step, Eh/bohr
    fmax_per_image: dict = field(default_factory=dict)  # last step, {image: max |F_perp|}

    scratch_dir: str = ''

    @property
    def energies(self):
        return self.path.energies if self.path is not None else None

    @property
    def highest_image(self):
        return int(np.argmax(self.energies))

    @property
    def barrier_kcalmol(self):
        e = self.energies
        return float((np.max(e) - e[0]) * const.hartree_to_kcalmol)

    # [TODO] the printing should be the same as when the program runs.
    def summary(self):
        """Lean output."""
        status = 'converged' if self.converged else 'NOT converged'
        lines = [f"{self.method}: {status} after {self.n_steps} steps"]
        if self.fmax_history:
            lines.append(f"  {'max|F_perp|:':<18s}{self.fmax_history[-1]:.2e} Eh/bohr")
            lines.append(f"  {'rms|F_perp|:':<18s}{self.frms_history[-1]:.2e} Eh/bohr")
        if self.path is not None and self.energies is not None:
            lines.append(f"  {'Images:':<18s}{self.path.n_images}")
            lines.append(f"  {'Barrier:':<18s}{self.barrier_kcalmol:.4f} kcal/mol "
                         f"(highest image: {self.highest_image})")
        return '\n'.join(lines)

    def report(self):
        """Full output."""
        print(self.summary())
        if self.fmax_per_image:
            print("  Last step, max|F_perp| per image:")
            for i in sorted(self.fmax_per_image):
                print(f"    image {i:3d}: {self.fmax_per_image[i]:.2e}")

    def __str__(self):
        return self.summary()

    def __repr__(self):
        n = self.path.n_images if self.path is not None else '?'
        return (f"NebResult(converged={self.converged}, steps={self.n_steps}, "
                f"images={n})")
