"""
This is the class for all the variables that vary along the reaction path.
The methods to alter this reaction path are also part of it, as it loops
over all the parts of the reaction path and keep the transformations 
consistent. This is important as the transformations are different for
each array. The class now reverses, clips and validates the path.

Categories
-------------------------
The arrays have different shapes, and they are transformed differently
when clipping or reversing:

  PER_IMAGE   one geometry per point along the path (len(N)).
              reverse [::-1] 
              clip [start:end]

  INTER_IMAGE one point per step (len(N)-1), because the displacement is
              built as differences (build_ds).
              reverse [::-1] | N-1 elements 
              clip drops x elements at each end

  SIGNED      reaction path. To be reersed it needs the order to be reversed
              and to have its sign flipped.

  INVARIANT   properties of the molecule, not of the path: masses, atomic numbers

Anything not listed in one of the four sets raises an exception at import time.
"""

from dataclasses import dataclass, field, fields
import numpy as np


PER_IMAGE = {
    'raw_coord', 'raw_coord_ang', 'aligned_coord', 'mass_scaled_coord',
    'energies', 'proj_energies', 'projected_corrections',
    'projected_frequencies', 'path_frequencies',
    'gradients', 'mw_gradients', 'hessians', 'mw_hessians',
    'normal_modes', 'proj_coordinates', 'tangents', 'gradient_rms',
    'center_of_mass'
}

INTER_IMAGE = {
    'displacement_per_step', 'displacement_non_mw','rotational_matrices', 
    'transl_vectors', 'alignement_rmsd'
}

SIGNED = {
    'reaction_path','cartesian_disp_non_mw'
}

INVARIANT = {
    'masses', 'atomic_numbers', 'direction','reactant_file'
}


def _empty():
    return np.array([])


@dataclass(eq=False)   # eq=False: we don't want __eq__ to generate 
                       # element-wise comparison
class ReactionPath:
    """One reaction path: geometries, energies, and everything derived."""

    # INVARIANT
    masses:         np.ndarray | None = None
    atomic_numbers: np.ndarray | None = None
    reactant_file:  str        | None = None

    # PER_IMAGE geometriews
    raw_coord:         np.ndarray | None = None
    raw_coord_ang:     np.ndarray | None = None
    aligned_coord:     np.ndarray | None = None
    mass_scaled_coord: np.ndarray | None = None

    # PER_IMAGE energies (they have to be transformed with the path)
    energies:              np.ndarray | None = None
    proj_energies:         np.ndarray | None = None
    projected_corrections: np.ndarray | None = None

    # PER_IMAGE derivatives
    gradients:    np.ndarray | None = None
    mw_gradients: np.ndarray | None = None
    hessians:     np.ndarray | None = None
    mw_hessians:  np.ndarray | None = None

    # PER_IMAGE vibrational information (projected and non projected)
    normal_modes:          np.ndarray | None = None
    projected_frequencies: np.ndarray | None = None
    path_frequencies:      np.ndarray | None = None
    proj_coordinates:      np.ndarray | None = None

    # SIGNED path geometry
    reaction_path:         np.ndarray | None = None  # signed
    cartesian_disp_non_mw: np.ndarray | None = None
    tangents:              np.ndarray | None = None
    gradient_rms:          np.ndarray | None = None

    # INTER_IMAGE displacements (length N-1 !!!!!) 
    displacement_per_step: np.ndarray | None = None
    displacement_non_mw:   np.ndarray | None = None

    # this proved to be useful, when working with the path
    # it can also be used later to automatically cehcking the thermodynamics
    # and keeping the path exotermic as default
    direction: str = 'forward'

    center_of_mass:      np.ndarray | None = None  # PER_IMAGE

    # INTER_IMAGE align_coordinates skips n == 0
    rotational_matrices: np.ndarray | None = None  # The correct backward transform is the transpose of R
                                                                    # but it is only for diagnostics so yeah... change if necessary
    transl_vectors:      np.ndarray | None = None
    alignement_rmsd:     np.ndarray | None = None

    ########################################################################### 


    def __post_init__(self): # runs right after __init__
        """
        Validating the path to avoid downstrem problems...
        """

        # Avoids previous errors where a list was added instead of a numpy array (may mask errors elsewhere '-.-)
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, (list, tuple)):
                setattr(self, f.name, np.asarray(val))

        n = self.n_images
        if n == 0:
            return

        for f in fields(self):
            name = f.name
            val = getattr(self, name)
            if name in INVARIANT or not isinstance(val, np.ndarray): # if it's not an array, no problem
                continue
            if val.size == 0:
                continue
            if name in PER_IMAGE or name in SIGNED:
                if len(val) != n:
                    raise ValueError(
                        f"{name} has {len(val)} points but the path has {n} images")
            elif name in INTER_IMAGE:
                if len(val) != n - 1:
                    raise ValueError(
                        f"{name} has {len(val)} points but it should have {n - 1}")

        if self.masses is not None and self.masses.size and self.raw_coord is not None and self.raw_coord.size:
            n_atoms = np.asarray(self.raw_coord[0]).shape[0]
            if len(self.masses) != n_atoms:
                raise ValueError(
                    f"There are {len(self.masses)} masses but {n_atoms} atoms! Not consistent!!")

    # properties
    # they make life so much easier...
    @property
    def n_images(self):
        for name in ('reaction_path', 'energies', 'raw_coord',
                     'proj_energies', 'mass_scaled_coord'): # start with reaction path and energies.
                                                            # if they are not built yet, try other arrays
            val = getattr(self, name)
            if isinstance(val, np.ndarray) and val.size:
                return len(val)
        return 0

    @property
    def n_atoms(self):
        if self.masses is not None and self.masses.size:
            return len(self.masses)
        if self.raw_coord.size: # VERY IMPORTANT! masses may not be loaded yet but the coordinates are
            return np.asarray(self.raw_coord[0]).shape[0]
        return 0

    # very useful
    def populated(self):
        """Names of the fields that actually hold data"""
        out = []
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, np.ndarray) and val.size:
                out.append(f.name)
            elif not isinstance(val, np.ndarray) and val is not None:
                out.append(f.name)
        return out

    def __repr__(self): # helps debigging
        return (f"ReactionPath({self.n_images} images, {self.n_atoms} atoms, "
                f"{self.direction}, and the following fiels are populated:\n{self.populated()})")

    # The reverse and clipping functions. Return a new path.
    # This is a design choice to avoid mutating the path and 
    # another caller mutating it again. Be careful with consistency
    # downstream. Also 'path' and 'path.reversed()' can be compared side by side.

    def reversed(self):
        """Returns the reversed path."""
        out = {}
        for f in fields(self):
            name = f.name
            val = getattr(self, name)

            if name == 'direction':
                out[name] = 'reverse' if val == 'forward' else 'forward'
            elif name in INVARIANT:
                out[name] = val
            elif not isinstance(val, np.ndarray) or val.size == 0:
                out[name] = val
            elif name in SIGNED:
                out[name] = -val[::-1]
            else:                       # PER_IMAGE and INTER_IMAGE alike
                out[name] = val[::-1]
        return ReactionPath(**out)

    def clipped(self, clip_start=0, clip_end=0):
        """
        Drop clip_start images from the front and clip_end from the back.
        Inter-image arrays lose one fewer element at each end, since they
        describe the gaps between images rather than the images themselves.
        """
        n = self.n_images
        if clip_start < 0 or clip_end < 0:
            raise ValueError("clip amounts must be non-negative")
        if clip_start + clip_end >= n:
            raise ValueError(
                f"clipping {clip_start} + {clip_end} would leave nothing "
                f"of a {n}-image path")

        stop = n - clip_end
        out = {}
        for f in fields(self):
            name = f.name
            val = getattr(self, name)

            if name in INVARIANT or not isinstance(val, np.ndarray) \
                    or val.size == 0:
                out[name] = val
            elif name in INTER_IMAGE:
                # displacements have n-1 elements and we are counting
                # from teh beginning of the array not the end
                out[name] = val[clip_start:stop - 1].copy()
            else:
                out[name] = val[clip_start:stop].copy()
        return ReactionPath(**out)

 
    def save(self, filename):
        """Write to a .npz so a path can be reused without loaded files."""
        data = {}
        for f in fields(self):
            val = getattr(self, f.name)
            data[f.name] = val if isinstance(val, np.ndarray) else np.array(val)
        np.savez_compressed(filename, **data)

    @classmethod
    def load(cls, filename):
        loaded = np.load(filename, allow_pickle=True)
        known = {f.name for f in fields(cls)}
        kwargs = {}
        for key in loaded.files:
            if key not in known:
                continue
            val = loaded[key]
            if key == 'direction':
                val = str(val)
            kwargs[key] = val
        return cls(**kwargs)


# Runs once at import. A field added to the dataclass without being assigned a
# category fails here immediately, instead of being silently skipped by
# reversed() and clipped() the way raw_coord_ang currently is.

_categorised = PER_IMAGE | INTER_IMAGE | SIGNED | INVARIANT
_declared = {f.name for f in fields(ReactionPath)}

_uncategorised = _declared - _categorised
if _uncategorised:
    raise RuntimeError(
        "ReactionPath fields with no transformation category: "
        f"{sorted(_uncategorised)}. Add each to PER_IMAGE, INTER_IMAGE, "
        "SIGNED or INVARIANT.")

_unknown = _categorised - _declared
if _unknown:
    raise RuntimeError(
        "Categorised names that are not ReactionPath fields: "
        f"{sorted(_unknown)}. Remove them or add the field.")
