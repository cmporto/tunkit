#!/usr/bin/env python3
"""
Stand-in for the ORCA executable, used to record v14 runs without ORCA.

Called exactly like ORCA by core.engine.run_orca_parallel / v14's _neb_run_parallel:
    fake_orca.py image_003.inp        (cwd = the image's scratch folder)

Reads the xyz block of the .inp (Angstrom), evaluates model_potential, and
writes image_003.engrad in ORCA's format, which orca_engrad_parser reads.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_potential  # noqa: E402

BOHR_TO_ANG = 0.529177210544
Z = {'N': 7, 'H': 1}


def read_xyz_block(inp_path):
    symbols, coords, inside = [], [], False
    with open(inp_path) as fo:
        for line in fo:
            s = line.strip()
            if s.lower().startswith('* xyz'):
                inside = True
                continue
            if inside:
                if s == '*':
                    break
                parts = s.split()
                symbols.append(parts[0])
                coords.append([float(x) / BOHR_TO_ANG for x in parts[1:4]])
    return symbols, coords


def main():
    inp = sys.argv[1]
    symbols, coords = read_xyz_block(inp)
    energy, grad = model_potential.energy_and_gradient(coords)
    stem = os.path.splitext(inp)[0]
    with open(stem + '.engrad', 'w') as fo:
        fo.write("#\n# Number of atoms\n#\n")
        fo.write(f" {len(symbols)}\n")
        fo.write("#\n# The current total energy in Eh\n#\n")
        fo.write(f"  {energy:.12f}\n")
        fo.write("#\n# The current gradient in Eh/bohr\n#\n")
        for g in grad.flatten():
            fo.write(f"  {g: .12f}\n")
        fo.write("#\n# The atomic numbers and current coordinates in Bohr\n#\n")
        for s, c in zip(symbols, coords):
            fo.write(f"  {Z[s]:3d} {c[0]: .7f} {c[1]: .7f} {c[2]: .7f}\n")
    print("                             ****ORCA TERMINATED NORMALLY**** (fake)")


if __name__ == '__main__':
    main()
