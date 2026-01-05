#!/bin/bash
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
