"""
FHI-aims calculation and queue templates.
Keep naming aligned with VASP templates for UI reuse (geometry_opt/static).
"""
import os
from typing import Dict, Optional


class AimsTemplates:
    CONTROL_GEOM_OPT = """# FHI-aims control.in (geometry optimization)
xc                 {xc}
relax_geometry     bfgs {fmax}
charge             {charge}
spin               none
sc_accuracy_rho    {acc_rho}
sc_accuracy_eev    {acc_eev}
sc_accuracy_etot   {acc_etot}
sc_iter_limit      {sc_iter}
k_grid             {k1} {k2} {k3}
compute_forces     .true.
output cube dens    .false.
"""

    CONTROL_STATIC = """# FHI-aims control.in (single point)
xc                 {xc}
charge             {charge}
spin               none
sc_accuracy_rho    {acc_rho}
sc_accuracy_eev    {acc_eev}
sc_accuracy_etot   {acc_etot}
sc_iter_limit      {sc_iter}
k_grid             {k1} {k2} {k3}
compute_forces     .false.
output cube dens    .false.
"""

    TEMPLATES = {
        "geometry_opt": CONTROL_GEOM_OPT,
        "static": CONTROL_STATIC,
    }

    @staticmethod
    def get_template_names():
        return list(AimsTemplates.TEMPLATES.keys())

    @staticmethod
    def get_template(template_name: str) -> Optional[str]:
        base_dir = os.path.dirname(__file__)
        fp = os.path.join(base_dir, "templates", f"control.{template_name}")
        if os.path.exists(fp):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                pass
        return AimsTemplates.TEMPLATES.get(template_name)

    @staticmethod
    def get_template_description(template_name: str) -> str:
        desc = {
            "geometry_opt": "FHI-aims Geometry Optimization",
            "static": "FHI-aims Single Point",
        }
        return desc.get(template_name, "Unknown")


class AimsQueueTemplates:
    SLURM = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={n_nodes}
#SBATCH --ntasks-per-node={n_procs}
#SBATCH --time={time_limit}
#SBATCH --partition={partition}
{email_directive}

module load aims 2>/dev/null || module load FHIaims 2>/dev/null || echo "aims module not loaded"
export OMP_NUM_THREADS=1
export TMPDIR=/tmp

cd $SLURM_SUBMIT_DIR
mpirun -np {total_procs} {aims_command} > aims.out 2>&1
"""

    TEMPLATES = {"slurm": AimsQueueTemplates.SLURM}

    @staticmethod
    def get_queue_systems():
        return list(AimsQueueTemplates.TEMPLATES.keys())

    @staticmethod
    def get_template(system_name: str) -> Optional[str]:
        base_dir = os.path.dirname(__file__)
        fp = os.path.join(base_dir, "templates", f"{system_name}.sh")
        if os.path.exists(fp):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                pass
        return AimsQueueTemplates.TEMPLATES.get(system_name)

    @staticmethod
    def get_template_description(system_name: str) -> str:
        desc = {"slurm": "SLURM (aims)"}
        return desc.get(system_name, "Unknown")
