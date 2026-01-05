#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks}
#SBATCH --time={time_limit}
#SBATCH --partition={partition}
{email_directive}

# ===== User-defined payload =====
{command}

echo "[$(date)] Job {job_name} finished"
