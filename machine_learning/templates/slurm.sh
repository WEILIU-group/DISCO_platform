#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks}
#SBATCH --time={time_limit}
#SBATCH --partition={partition}

python train_script.py
echo "[$(date)] Job {job_name} finished"


