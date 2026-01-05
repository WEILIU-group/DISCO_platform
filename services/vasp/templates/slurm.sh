#!/bin/bash
#SBATCH --job-name=vasp
#SBATCH --output=out.%j
#SBATCH --error=err.%j
#SBATCH --partition=vasp
#SBATCH --nodes=1
#SBATCH --exclusive

# generate POTCAR
vaspkit << EOF
103
EOF

source /Public/home/zyy/intel/oneapi/setvars.sh intel64 > /dev/null 2>&1
export PATH=/Public/home/zyy/soft/vasp.6.3.2_c/bin:$PATH

mpirun -np $(nproc)  vasp_gam >> vasp.log 2>&1