import os
import time
import numpy as np
import traceback
import subprocess
from pathlib import Path
from ase.build import fcc111, add_adsorbate
from ase.optimize import BFGS
from ase.io import write, read
from ase.constraints import FixAtoms

BASE_DIR = Path(__file__).resolve().parents[1]
LOCAL_REMOTE_CALC = BASE_DIR / "temp_remote_calc.py"

# 确保输出目录存在
OUTPUT_DIR = "simulation_outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Remote Configuration
REMOTE_HOST = "116.142.76.181"
REMOTE_PORT = "16445"
REMOTE_USER = "root"
REMOTE_TMP_DIR = "/tmp"
CONTAINER_NAME = "pytorch270"
CONTAINER_SCRIPT_PATH = "/root/remote_calc.py"

def run_ssh_command(command):
    """Run a command on the remote server via SSH."""
    ssh_cmd = ["ssh", "-p", REMOTE_PORT, f"{REMOTE_USER}@{REMOTE_HOST}", command]
    print(f"  [SSH] Executing: {' '.join(ssh_cmd)}")
    result = subprocess.run(ssh_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [SSH] Error: {result.stderr}")
        raise Exception(f"SSH command failed: {result.stderr}")
    return result.stdout

def copy_to_remote(local_path, remote_path):
    """Copy a file to the remote server via SCP."""
    scp_cmd = ["scp", "-P", REMOTE_PORT, local_path, f"{REMOTE_USER}@{REMOTE_HOST}:{remote_path}"]
    print(f"  [SCP] Uploading: {local_path} -> {remote_path}")
    result = subprocess.run(scp_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [SCP] Error: {result.stderr}")
        raise Exception(f"SCP upload failed: {result.stderr}")

def copy_from_remote(remote_path, local_path):
    """Copy a file from the remote server via SCP."""
    scp_cmd = ["scp", "-P", REMOTE_PORT, f"{REMOTE_USER}@{REMOTE_HOST}:{remote_path}", local_path]
    print(f"  [SCP] Downloading: {remote_path} -> {local_path}")
    result = subprocess.run(scp_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [SCP] Error: {result.stderr}")
        raise Exception(f"SCP download failed: {result.stderr}")

def real_structure_builder(surface_name: str, adsorbate: str, site: str):
    """
    真实的建模工具：使用 ASE 构建表面并添加吸附物。
    目前支持: Pt(111) + O
    """
    print(f"  [RealTool] 正在构建真实原子模型: {surface_name} + {adsorbate} @ {site}...")
    
    try:
        # 1. 构建表面 (以 Pt(111) 为例)
        if "Pt" in surface_name and "111" in surface_name:
            # 创建 3x3x4 的 slab，真空层 10A
            slab = fcc111('Pt', size=(3, 3, 4), vacuum=10.0)
        else:
            # 默认 fallback 到 Pt(111)
            print(f"  [RealTool] 警告: 暂不支持 {surface_name}，回退到 Pt(111)")
            slab = fcc111('Pt', size=(3, 3, 4), vacuum=10.0)

        # 2. 添加吸附物 (映射成 ASE 关键字)
        ase_site = "ontop"
        if "top" in site.lower(): ase_site = "ontop"
        elif "bridge" in site.lower(): ase_site = "bridge"
        elif "hollow" in site.lower():
            if "fcc" in site.lower(): ase_site = "fcc"
            elif "hcp" in site.lower(): ase_site = "hcp"
            else: ase_site = "fcc" # 默认 hollow -> fcc
            
        add_adsorbate(slab, adsorbate, height=1.8, position=ase_site)
        
        # 3. 保存文件
        filename = f"{OUTPUT_DIR}/STRUCT_{surface_name}_{adsorbate}_{site}.xyz"
        # 替换掉文件名里可能有的非法字符
        filename = filename.replace("(", "").replace(")", "")
        filename = filename.replace("/", "_")
        
        write(filename, slab)
        print(f"  [RealTool] 模型已保存至: {filename}")
        return filename
        
    except Exception as e:
        print(f"  [RealTool] 建模失败: {e}")
        return None

def real_energy_calculator(structure_path: str):
    """
    真实的计算工具：将结构上传至远端容器，调用 deepmd 优化脚本，返回能量。
    """
    print(f"  [RealTool] 上传并提交真实计算: {structure_path} ...")

    if not LOCAL_REMOTE_CALC.exists():
        raise FileNotFoundError(f"remote calc script missing: {LOCAL_REMOTE_CALC}")

    basename = os.path.basename(structure_path)
    remote_xyz = f"{REMOTE_TMP_DIR}/{basename}"
    remote_script = f"{REMOTE_TMP_DIR}/remote_calc.py"

    try:
        # 1) 上传结构和脚本到远端主机
        copy_to_remote(structure_path, remote_xyz)
        copy_to_remote(str(LOCAL_REMOTE_CALC), remote_script)

        # 2) 将文件复制进容器并执行
        exec_cmd = (
            f"docker cp {remote_script} {CONTAINER_NAME}:{CONTAINER_SCRIPT_PATH} && "
            f"docker cp {remote_xyz} {CONTAINER_NAME}:/root/{basename} && "
            f"docker exec {CONTAINER_NAME} python {CONTAINER_SCRIPT_PATH} /root/{basename}"
        )
        out = run_ssh_command(exec_cmd)

        # 3) 解析能量
        energy = None
        for line in out.splitlines():
            if "ENERGY_RESULT" in line:
                try:
                    energy = float(line.split(":")[-1])
                except Exception:
                    pass
        if energy is None:
            raise RuntimeError(f"未解析到能量, 输出: {out}")

        # 4) 下载优化结构
        opt_name = basename.replace(".xyz", "_opt.xyz")
        remote_opt = f"{REMOTE_TMP_DIR}/{opt_name}"
        # 将容器内优化文件拷到宿主再下载
        run_ssh_command(f"docker cp {CONTAINER_NAME}:/root/{opt_name} {remote_opt}")
        local_opt = os.path.join(OUTPUT_DIR, opt_name)
        copy_from_remote(remote_opt, local_opt)

        print(f"  [RealTool] 能量 = {energy:.4f} eV, 优化结构 -> {local_opt}")
        return energy
    except Exception as e:
        print(f"  [RealTool] 真实计算失败: {e}")
        traceback.print_exc()
        return None
