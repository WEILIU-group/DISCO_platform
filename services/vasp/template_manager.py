"""
VASP 计算模板管理系统
支持多种计算类型和队列系统
"""

import os
from typing import Dict, Optional, Tuple


class VaspCalculationTemplates:
    """VASP 计算类型模板库"""
    
    # 结构优化模板 (Geometry Optimization)
    GEOMETRY_OPTIMIZATION = """! Structure Relaxation / Geometry Optimization
! Relax atomic positions and lattice parameters

! === General Settings ===
ENCUT = {encut}          ! Plane-wave cutoff energy (eV)
NSW = {nsw}              ! Number of ionic steps
IBRION = {ibrion}        ! Ionic relaxation algorithm (2=CG, 1=RMM-DIIS)
POTIM = {potim}          ! Time step for ionic motion
EDIFF = {ediff}          ! Electronic convergence criterion (eV)
EDIFFG = -0.02           ! Ionic convergence criterion (forces in eV/Angstrom)
PREC = {prec}            ! Precision level (Low, Medium, High)
ISIF = 3                 ! Stress tensor calculation and lattice optimization

! === Electronic Structure ===
NELM = {nelm}            ! Max electronic iterations
NELMIN = {nelmin}        ! Min electronic iterations
ISMEAR = {ismear}        ! Smearing method (0=Gaussian, -5=tetrahedra)
SIGMA = {sigma}          ! Smearing width (eV)

! === Wave Function ===
ALGO = FAST              ! Algorithm (Fast, Normal, Accurate)
LREAL = Auto             ! Projection operators in real space

! === Output ===
LWAVE = .FALSE.          ! Write WAVECAR
LCHARG = .FALSE.         ! Write CHGCAR
LVTOT = .FALSE.          ! Write total potential
LORBIT = 11              ! Output magnetization
NPAR = 4                 ! Parallelization parameter
LASPH = .TRUE.           ! Aspherical Hartree potential
"""

    # 单点计算模板 (Static/Single Point Energy)
    STATIC_ENERGY = """! Static Energy Calculation / Single Point Energy
! Calculate energy of a fixed structure

! === General Settings ===
ENCUT = {encut}          ! Plane-wave cutoff energy (eV)
NSW = 0                  ! Number of ionic steps (0 = no relaxation)
IBRION = -1              ! No ionic movement
EDIFF = {ediff}          ! Electronic convergence criterion (eV)
PREC = {prec}            ! Precision level (Low, Medium, High)

! === Electronic Structure ===
NELM = {nelm}            ! Max electronic iterations
NELMIN = {nelmin}        ! Min electronic iterations
ISMEAR = {ismear}        ! Smearing method (0=Gaussian, -5=tetrahedra)
SIGMA = {sigma}          ! Smearing width (eV)

! === Wave Function ===
ALGO = FAST              ! Algorithm (Fast, Normal, Accurate)
LREAL = Auto             ! Projection operators in real space

! === Output ===
LWAVE = .FALSE.          ! Write WAVECAR
LCHARG = .TRUE.          ! Write charge density
LVTOT = .FALSE.          ! Write total potential
LORBIT = 11              ! Output magnetization
NPAR = 4                 ! Parallelization parameter
LASPH = .TRUE.           ! Aspherical Hartree potential
"""

    # 频率计算模板 (Frequency/Vibrational Analysis)
    FREQUENCY_ANALYSIS = """! Frequency Analysis / Vibrational Properties
! Calculate vibrational frequencies and normal modes

! === General Settings ===
ENCUT = {encut}          ! Plane-wave cutoff energy (eV)
NSW = 1                  ! Number of ionic steps (1 for frequency)
IBRION = 5               ! Frequency calculation (5=finite differences)
POTIM = 0.015            ! Finite difference step size
EDIFF = {ediff}          ! Electronic convergence criterion
PREC = {prec}            ! Precision level (High recommended)

! === Electronic Structure ===
NELM = {nelm}            ! Max electronic iterations
NELMIN = {nelmin}        ! Min electronic iterations
ISMEAR = 0               ! Gaussian smearing for frequencies
SIGMA = 0.1              ! Smearing width

! === Symmetry & Output ===
NFREE = 2                ! Number of displacements (1 or 2)
LWAVE = .FALSE.          ! Write WAVECAR
LCHARG = .FALSE.         ! Write charge density
LORBIT = 11              ! Output magnetization
NPAR = 4                 ! Parallelization parameter
"""

    # 过渡态搜索模板 (Transition State / NEB)
    TRANSITION_STATE = """! Transition State / NEB Calculation
! Find reaction pathways using Nudged Elastic Band

! === General Settings ===
ENCUT = {encut}          ! Plane-wave cutoff energy (eV)
NSW = {nsw}              ! Number of ionic steps
IBRION = {ibrion}        ! Ionic relaxation (1=RMM-DIIS for NEB)
EDIFF = {ediff}          ! Electronic convergence criterion
PREC = {prec}            ! Precision level (High)
ISIF = 0                 ! No stress tensor for NEB

! === NEB Parameters ===
ICHAIN = 0               ! Chain job type (0=NEB)
IMAGES = 5               ! Number of intermediate images
SPRING = -5.0            ! Spring constant for NEB
LCLIMB = .TRUE.          ! Climbing image method

! === Electronic Structure ===
NELM = {nelm}            ! Max electronic iterations
NELMIN = {nelmin}        ! Min electronic iterations
ISMEAR = {ismear}        ! Smearing method
SIGMA = {sigma}          ! Smearing width

! === Output ===
LWAVE = .FALSE.          ! Write WAVECAR
LCHARG = .FALSE.         ! Write charge density
LORBIT = 11              ! Output magnetization
"""

    # 吸附体系模板 (Adsorbate / Surface + Adsorbate)
    ADSORBATE_SYSTEM = """! Adsorbate System / Surface Chemistry
! Calculate adsorption energy and geometry

! === General Settings ===
ENCUT = {encut}          ! Plane-wave cutoff energy (eV)
NSW = {nsw}              ! Number of ionic steps
IBRION = {ibrion}        ! Ionic relaxation (2=CG)
POTIM = {potim}          ! Time step
EDIFF = {ediff}          ! Electronic convergence criterion
EDIFFG = -0.02           ! Ionic convergence criterion
PREC = {prec}            ! Precision level (High)
ISIF = 2                 ! Relax atomic positions only (keep cell fixed)

! === Electronic Structure ===
NELM = {nelm}            ! Max electronic iterations
NELMIN = {nelmin}        ! Min electronic iterations
ISMEAR = {ismear}        ! Smearing method
SIGMA = {sigma}          ! Smearing width
LOPTICS = .FALSE.        ! Optical properties

! === Dipole Correction ===
LDIPOL = .TRUE.          ! Dipole moment correction
IDIPOL = 3               ! Direction for dipole correction
DIPOL = 0.5 0.5 0.5      ! Dipole center

! === Output ===
LWAVE = .FALSE.          ! Write WAVECAR
LCHARG = .FALSE.         ! Write charge density
LORBIT = 11              ! Output magnetization
"""

    TEMPLATES_DICT = {
        "geometry_opt": GEOMETRY_OPTIMIZATION,
        "static": STATIC_ENERGY,
        "frequency": FREQUENCY_ANALYSIS,
        "transition_state": TRANSITION_STATE,
        "adsorbate": ADSORBATE_SYSTEM,
    }

    @staticmethod
    def get_template_names() -> list:
        """获取所有可用的模板名称"""
        return list(VaspCalculationTemplates.TEMPLATES_DICT.keys())

    @staticmethod
    def get_template(template_name: str) -> Optional[str]:
        """获取指定模板内容"""
        # 优先从磁盘模板文件读取（服务目录下的 templates/INCAR.<name>）
        base_dir = os.path.dirname(__file__)
        file_path = os.path.join(base_dir, "templates", f"INCAR.{template_name}")
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception:
                # 读取失败则回退到内置模板
                pass

        return VaspCalculationTemplates.TEMPLATES_DICT.get(template_name)

    @staticmethod
    def get_template_description(template_name: str) -> str:
        """获取模板描述"""
        descriptions = {
            "geometry_opt": "Structure Relaxation / Geometry Optimization",
            "static": "Static Energy Calculation / Single Point",
            "frequency": "Vibrational Analysis / Frequency",
            "transition_state": "Transition State / NEB Pathway",
            "adsorbate": "Adsorbate System / Surface Chemistry",
        }
        return descriptions.get(template_name, "Unknown")


class QueueSystemTemplates:
    """队列系统脚本模板库"""
    
    # SLURM 队列系统模板
    SLURM = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={n_nodes}
#SBATCH --ntasks-per-node={n_procs}
#SBATCH --time={time_limit}
#SBATCH --partition={partition}
{email_directive}
# ===== Job Submission Environment Setup =====
module load VASP/6.3.2-GPU 2>/dev/null || module load vasp 2>/dev/null || echo "VASP module not loaded"

export OMP_NUM_THREADS=1
export TMPDIR=/tmp

# ===== Job Execution =====
echo "=== Job Submitted at $(date) ===" >> job.log
echo "Host: $(hostname)" >> job.log
echo "Total Processes: {total_procs}" >> job.log

mpirun -np {total_procs} {vasp_command} >> vasp.log 2>&1
EXIT_CODE=$?

echo "=== Job Completed at $(date), Exit Code: $EXIT_CODE ===" >> job.log

exit $EXIT_CODE
"""

    # PBS 队列系统模板
    PBS = """#!/bin/bash
#PBS -N {job_name}
#PBS -l nodes={n_nodes}:ppn={n_procs}
#PBS -l walltime={time_limit}
#PBS -q {partition}
{email_directive}
# ===== Job Submission Environment Setup =====
module load VASP/6.3.2 2>/dev/null || echo "VASP module not loaded"

export OMP_NUM_THREADS=1
export TMPDIR=/tmp

# Create temporary directory for job
cd $PBS_O_WORKDIR

# ===== Job Execution =====
echo "=== Job Submitted at $(date) ===" >> job.log
echo "Host: $(hostname)" >> job.log
echo "Total Processes: {total_procs}" >> job.log

mpirun -np {total_procs} {vasp_command} >> vasp.log 2>&1
EXIT_CODE=$?

echo "=== Job Completed at $(date), Exit Code: $EXIT_CODE ===" >> job.log

exit $EXIT_CODE
"""

    # SGE (Sun Grid Engine) 模板
    SGE = """#!/bin/bash
#$ -N {job_name}
#$ -pe mpi {total_procs}
#$ -l h_rt={time_limit}
#$ -q {partition}
{email_directive}
# ===== Job Submission Environment Setup =====
module load VASP/6.3.2 2>/dev/null || echo "VASP module not loaded"

export OMP_NUM_THREADS=1
export TMPDIR=/tmp

# ===== Job Execution =====
echo "=== Job Submitted at $(date) ===" >> job.log
echo "Host: $(hostname)" >> job.log
echo "Total Processes: {total_procs}" >> job.log

mpirun -np {total_procs} {vasp_command} >> vasp.log 2>&1
EXIT_CODE=$?

echo "=== Job Completed at $(date), Exit Code: $EXIT_CODE ===" >> job.log

exit $EXIT_CODE
"""

    TEMPLATES_DICT = {
        "slurm": SLURM,
        "pbs": PBS,
        "sge": SGE,
    }

    @staticmethod
    def get_queue_systems() -> list:
        """获取所有支持的队列系统"""
        return list(QueueSystemTemplates.TEMPLATES_DICT.keys())

    @staticmethod
    def get_template(system_name: str) -> Optional[str]:
        """获取指定队列系统的模板"""
        # 优先从磁盘模板文件读取（服务目录下的 templates/<system_name>.sh）
        base_dir = os.path.dirname(__file__)
        file_path = os.path.join(base_dir, "templates", f"{system_name}.sh")
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception:
                # 读取失败则回退到内置模板
                pass

        return QueueSystemTemplates.TEMPLATES_DICT.get(system_name)

    @staticmethod
    def get_template_description(system_name: str) -> str:
        """获取队列系统描述"""
        descriptions = {
            "slurm": "SLURM Workload Manager",
            "pbs": "PBS / OpenPBS",
            "sge": "Sun Grid Engine (SGE)",
        }
        return descriptions.get(system_name, "Unknown")


class SessionTemplateEditor:
    """会话级模板编辑器 - 修改只对当前会话有效"""
    
    def __init__(self):
        """初始化会话编辑器"""
        self.vasp_templates = {}  # 存储会话级修改的VASP模板
        self.queue_templates = {}  # 存储会话级修改的队列模板
        
    def get_vasp_template(self, template_name: str) -> str:
        """
        获取 VASP 模板（优先返回会话修改版本）
        
        Args:
            template_name: 模板名称
            
        Returns:
            模板内容
        """
        if template_name in self.vasp_templates:
            return self.vasp_templates[template_name]
        return VaspCalculationTemplates.get_template(template_name)
    
    def get_queue_template(self, queue_system: str) -> str:
        """
        获取队列模板（优先返回会话修改版本）
        
        Args:
            queue_system: 队列系统名称
            
        Returns:
            模板内容
        """
        if queue_system in self.queue_templates:
            return self.queue_templates[queue_system]
        return QueueSystemTemplates.get_template(queue_system)
    
    def update_vasp_template(self, template_name: str, content: str) -> Tuple[bool, str]:
        """
        更新 VASP 模板（仅当前会话）
        
        Args:
            template_name: 模板名称
            content: 新的模板内容
            
        Returns:
            (成功标志, 消息)
        """
        if template_name not in VaspCalculationTemplates.TEMPLATES_DICT:
            return False, f"模板不存在: {template_name}"
        
        self.vasp_templates[template_name] = content
        return True, f"✓ VASP 模板已更新（会话级）: {template_name}"
    
    def update_queue_template(self, queue_system: str, content: str) -> Tuple[bool, str]:
        """
        更新队列模板（仅当前会话）
        
        Args:
            queue_system: 队列系统名称
            content: 新的模板内容
            
        Returns:
            (成功标志, 消息)
        """
        if queue_system not in QueueSystemTemplates.TEMPLATES_DICT:
            return False, f"队列系统不存在: {queue_system}"
        
        self.queue_templates[queue_system] = content
        return True, f"✓ 队列模板已更新（会话级）: {queue_system}"
    
    def get_vasp_template_with_params(self, template_name: str, params: Dict) -> str:
        """
        获取 VASP 模板并替换参数
        
        Args:
            template_name: 模板名称
            params: 参数字典
            
        Returns:
            带有替换参数的模板内容
        """
        template = self.get_vasp_template(template_name)
        content = template
        
        for key, value in params.items():
            content = content.replace(f"{{{key}}}", str(value))
        
        return content
    
    def get_queue_template_with_params(self, queue_system: str, params: Dict) -> str:
        """
        获取队列模板并替换参数
        
        Args:
            queue_system: 队列系统名称
            params: 参数字典
            
        Returns:
            带有替换参数的模板内容
        """
        template = self.get_queue_template(queue_system)
        content = template
        
        for key, value in params.items():
            content = content.replace(f"{{{key}}}", str(value))
        
        return content
    
    def reset_all(self):
        """重置所有会话级修改"""
        self.vasp_templates.clear()
        self.queue_templates.clear()
        return True, "✓ 所有会话级模板已重置"


# 全局会话编辑器实例
SESSION_EDITOR = SessionTemplateEditor()
