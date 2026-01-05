"""
VASP 工作流管理器
负责生成完整的VASP计算工作流：输入文件、队列脚本、远程提交、结果获取
"""

import time
import datetime
from typing import Dict, Tuple, List, Optional
from pathlib import Path
import json

from .config import VaspConfigManager, VaspInputFileGenerator, VaspResultParser
from ..remote_server import SSHManager


class VaspWorkflowManager:
    """VASP 工作流管理 - 集成所有VASP计算步骤"""

    def __init__(self, config: Dict):
        """
        初始化工作流管理器

        Args:
            config: 配置字典，包含:
                - remote_server: SSH配置
                - remote_paths: 远程路径配置
                - vasp_defaults: VASP默认参数
                - queue_system: 队列系统配置
        """
        self.config = config
        self.vasp_config = VaspConfigManager()
        self.result_parser = VaspResultParser()
        self.ssh = None
        self.current_job_id = None
        self.current_remote_dir = None

    def connect_remote(self) -> Tuple[bool, str]:
        """连接到远程服务器"""
        try:
            ssh_config = self.config.get("remote_server", {})
            self.ssh = SSHManager(**ssh_config)
            success, msg = self.ssh.connect()
            if success:
                self.ssh.open_sftp()
            return success, msg
        except Exception as e:
            return False, f"连接失败: {str(e)}"

    def disconnect_remote(self):
        """断开远程连接"""
        if self.ssh:
            self.ssh.close()

    def generate_slurm_script(
        self,
        job_name: str = "VASP_Job",
        n_nodes: int = 1,
        n_procs: int = 16,
        time_limit: str = "00:30:00",
        partition: str = "gpu",
        email: Optional[str] = None,
        vasp_command: str = "vasp_std"
    ) -> str:
        """
        生成SLURM队列提交脚本（从模板生成）

        Args:
            job_name: 任务名称
            n_nodes: 计算节点数
            n_procs: 每个节点的进程数
            time_limit: 计算时间限制
            partition: 分区名称
            email: 通知邮箱
            vasp_command: VASP执行命令

        Returns:
            SLURM脚本内容
        """
        return self.vasp_config.generate_slurm_script(
            job_name=job_name,
            n_nodes=n_nodes,
            n_procs=n_procs,
            time_limit=time_limit,
            partition=partition,
            email=email,
            vasp_command=vasp_command
        )

    def prepare_calculation(
        self,
        structure_dict: Dict,
        structure_name: str,
        vasp_params: Optional[Dict] = None,
        slurm_params: Optional[Dict] = None,
        incar_content: Optional[str] = None,
        slurm_content: Optional[str] = None
    ) -> Tuple[bool, Dict]:
        """
        准备VASP计算文件

        Args:
            structure_dict: pymatgen Structure.as_dict() 结果
            structure_name: 结构名称/标识符
            vasp_params: VASP计算参数（覆盖默认值）
            slurm_params: SLURM脚本参数（覆盖默认值）
            incar_content: 直接提供的INCAR内容（如果提供，将跳过模板生成）
            slurm_content: 直接提供的SLURM脚本内容（如果提供，将跳过模板生成）

        Returns:
            (成功标志, 文件字典 {'poscar': ..., 'incar': ..., 'kpoints': ..., 'slurm': ...})
        """
        try:
            # 合并参数
            final_vasp_params = self.config.get("vasp_defaults", {}).copy()
            if vasp_params:
                final_vasp_params.update(vasp_params)

            final_slurm_params = {
                "job_name": f"VASP_{structure_name}",
                "n_nodes": 1,
                "n_procs": 16,
                "time_limit": "01:00:00",
                "partition": "gpu",
                "vasp_command": "vasp_std"
            }
            if slurm_params:
                final_slurm_params.update(slurm_params)

            # 生成POSCAR
            poscar = VaspInputFileGenerator.generate_poscar(structure_dict, comment=structure_name)

            # 生成INCAR
            if incar_content:
                incar = incar_content
            else:
                incar = self.vasp_config.generate_incar(final_vasp_params)

            # 生成KPOINTS
            kmesh = final_vasp_params.get("kmesh", [4, 4, 4])
            kpoints = self.vasp_config.generate_kpoints(tuple(kmesh))

            # 生成SLURM脚本
            if slurm_content:
                slurm_script = slurm_content
            else:
                slurm_script = self.generate_slurm_script(**final_slurm_params)

            return True, {
                "poscar": poscar,
                "incar": incar,
                "kpoints": kpoints,
                "slurm": slurm_script,
                "params": {
                    "structure_name": structure_name,
                    "vasp_params": final_vasp_params,
                    "slurm_params": final_slurm_params
                }
            }

        except Exception as e:
            return False, {"error": str(e)}

    def submit_calculation(
        self,
        calculation_files: Dict,
        use_default_potcar: bool = False,
        potcar_path: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        提交VASP计算到远程服务器

        Args:
            calculation_files: prepare_calculation() 返回的文件字典
            use_default_potcar: 是否使用默认POTCAR
            potcar_path: 远程POTCAR文件路径

        Returns:
            (成功标志, 远程工作目录 或 错误信息)
        """
        if not self.ssh:
            return False, "未连接到远程服务器，请先调用 connect_remote()"

        try:
            structure_name = calculation_files["params"]["structure_name"]
            remote_base = self.config.get("remote_paths", {}).get("vasp_base", "VASP_JOBS")

            # 创建远程工作目录
            timestamp = int(time.time())
            self.current_remote_dir = f"{remote_base}/VASP_JOB_{structure_name}_{timestamp}"
            success, msg = self.ssh.mkdir_remote(self.current_remote_dir)
            if not success:
                return False, f"创建目录失败: {msg}"

            # 上传文件
            for filename, content in calculation_files.items():
                if filename == "params":
                    continue
                remote_path = f"{self.current_remote_dir}/{filename.upper()}"
                success, msg = self.ssh.write_remote_file(remote_path, content)
                if not success:
                    return False, f"上传 {filename} 失败: {msg}"

            # 处理POTCAR
            if use_default_potcar and potcar_path:
                ret_code, out, err = self.ssh.exec_command(
                    f"cp {potcar_path} {self.current_remote_dir}/POTCAR"
                )
                if ret_code != 0:
                    return False, f"复制POTCAR失败: {err}"

            # 上传并提交SLURM脚本
            ret_code, out, err = self.ssh.exec_command(
                f"cd {self.current_remote_dir} && chmod +x SLURM && sbatch SLURM"
            )

            if ret_code == 0 and "Submitted batch job" in out:
                job_id = out.split()[-1]
                self.current_job_id = job_id
                return True, job_id
            else:
                return False, f"提交任务失败: {out} {err}"

        except Exception as e:
            return False, str(e)

    def poll_and_get_results(
        self,
        job_id: Optional[str] = None,
        check_interval: int = 5,
        max_wait_seconds: int = 1800,
        on_poll_callback=None
    ) -> Tuple[bool, Dict]:
        """
        等待任务完成并获取结果

        Args:
            job_id: 任务ID（如果为None则使用 current_job_id）
            check_interval: 轮询间隔（秒）
            max_wait_seconds: 最大等待时间（秒）
            on_poll_callback: 轮询回调函数

        Returns:
            (成功标志, 结果字典或错误信息)
        """
        if not self.ssh:
            return False, {"error": "未连接到远程服务器"}

        target_job_id = job_id or self.current_job_id
        if not target_job_id:
            return False, {"error": "未指定Job ID"}

        try:
            # 等待任务完成
            success, msg = self.ssh.poll_job_completion(
                target_job_id,
                check_interval=check_interval,
                max_wait_seconds=max_wait_seconds,
                on_poll_callback=on_poll_callback
            )

            if not success:
                return False, {"error": msg}

            # 获取结果文件
            remote_dir = self.current_remote_dir
            if not remote_dir:
                return False, {"error": "无法确定远程工作目录"}

            return self.get_calculation_results(remote_dir, target_job_id)

        except Exception as e:
            return False, {"error": str(e)}

    def get_calculation_results(self, remote_dir: str, job_id: str = None) -> Tuple[bool, Dict]:
        """
        获取计算结果（不等待，直接读取）
        
        Args:
            remote_dir: 远程工作目录
            job_id: 任务ID（可选）
            
        Returns:
            (成功标志, 结果字典)
        """
        if not self.ssh:
            return False, {"error": "未连接到远程服务器"}
            
        results = {}
        
        try:
            # 读取OSZICAR
            success, oszicar_content = self.ssh.read_remote_file(
                f"{remote_dir}/OSZICAR", max_bytes=5000
            )
            if success:
                parsed, oszicar_result = self.result_parser.parse_oszicar(oszicar_content)
                if parsed:
                    results["oszicar"] = oszicar_result

            # 读取OUTCAR
            success, outcar_content = self.ssh.read_remote_file(
                f"{remote_dir}/OUTCAR", max_bytes=10000
            )
            if success:
                results["outcar"] = self.result_parser.parse_outcar(outcar_content)

            # 读取计算日志
            success, log_content = self.ssh.read_remote_file(
                f"{remote_dir}/vasp.log", max_bytes=2000
            )
            if success:
                results["log"] = log_content

            results["remote_dir"] = remote_dir
            if job_id:
                results["job_id"] = job_id
            results["fetched_at"] = datetime.datetime.now().isoformat()

            return True, results
            
        except Exception as e:
            return False, {"error": str(e)}

    def download_output(
        self,
        local_dir: str,
        job_id: Optional[str] = None,
        files_to_download: Optional[List[str]] = None
    ) -> Tuple[bool, str]:
        """
        从远程服务器下载计算结果

        Args:
            local_dir: 本地保存目录
            job_id: 任务ID（可选）
            files_to_download: 要下载的文件列表（默认：OUTCAR, OSZICAR, vasprun.xml）

        Returns:
            (成功标志, 消息)
        """
        if not self.ssh:
            return False, "未连接到远程服务器"

        if not self.current_remote_dir:
            return False, "无法确定远程工作目录"

        if files_to_download is None:
            files_to_download = ["OUTCAR", "OSZICAR", "vasprun.xml"]

        try:
            Path(local_dir).mkdir(parents=True, exist_ok=True)

            for filename in files_to_download:
                remote_file = f"{self.current_remote_dir}/{filename}"
                local_file = str(Path(local_dir) / filename)

                success, msg = self.ssh.download_result_file(remote_file, local_file)
                if not success:
                    return False, f"下载 {filename} 失败: {msg}"

            return True, f"下载完成到 {local_dir}"

        except Exception as e:
            return False, str(e)

    def cleanup_remote(self, remote_dir: Optional[str] = None) -> Tuple[bool, str]:
        """
        清理远程工作目录

        Args:
            remote_dir: 要清理的目录（默认使用current_remote_dir）

        Returns:
            (成功标志, 消息)
        """
        if not self.ssh:
            return False, "未连接到远程服务器"

        target_dir = remote_dir or self.current_remote_dir
        if not target_dir:
            return False, "无法确定远程工作目录"

        return self.ssh.cleanup_remote_dir(target_dir)

