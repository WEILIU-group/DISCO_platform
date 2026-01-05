#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={n_nodes}
#SBATCH --ntasks-per-node={n_procs}
#SBATCH --time={time_limit}
#SBATCH --partition={partition}
{email_directive}
module load SISSO 2>/dev/null || echo "SISSO module not loaded"

export OMP_NUM_THREADS={omp_threads}

echo "=== SISSO Feature Selection Started at $(date) ===" >> sisso.log
mpirun -np {total_procs} sisso >> sisso.log 2>&1
EXIT_CODE=$?
echo "=== SISSO Feature Selection Finished at $(date), Exit Code: $EXIT_CODE ===" >> sisso.log

exit $EXIT_CODE
