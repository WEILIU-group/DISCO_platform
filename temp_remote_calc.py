
import sys
import os
import numpy as np
from ase.io import read, write
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from deepmd.calculator import DP

def run_calc(xyz_path):
    print(f"Processing {xyz_path}...")
    atoms = read(xyz_path)
    
    # Constraints
    z_positions = atoms.positions[:, 2]
    min_z = np.min(z_positions)
    mask = [atom.position[2] < min_z + 3.0 for atom in atoms]
    atoms.set_constraint(FixAtoms(mask=mask))
    
    model_path = "/root/share/work2/pre_trained_models/DPA-3.1-3M.pt"
    print(f"Loading model from {model_path}")
    
    try:
        atoms.calc = DP(model=model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    print("Starting optimization...")
    dyn = BFGS(atoms)
    dyn.run(fmax=0.05)
    
    e = atoms.get_potential_energy()
    print(f"ENERGY_RESULT:{e}")
    
    opt_path = xyz_path.replace(".xyz", "_opt.xyz")
    write(opt_path, atoms)
    print(f"Saved optimized structure to {opt_path}")

if __name__ == "__main__":
    run_calc(sys.argv[1])
