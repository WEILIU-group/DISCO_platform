"""
A minimal FHI-aims workflow manager: prepare geometry/control, submit via SSH/Slurm, fetch results.
"""
import io
import time
import datetime
import re
from typing import Dict, Tuple, List, Optional
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write, read

from .template_manager import AimsTemplates, AimsQueueTemplates
from ..remote_server import SSHManager


def _structure_to_geometry_in(structure_dict: Dict) -> str:
    s = Structure.from_dict(structure_dict)
    atoms = AseAtomsAdaptor.get_atoms(s)
    buf = io.StringIO()
    write(buf, atoms, format="aims")
    return buf.getvalue()


def _parse_aims_energy(out_content: str) -> Optional[float]:
    # Try common patterns (eV)
    patterns = [
        r"Total energy\s+of\s+the\s+DFT.*?:\s*([-+]?[0-9]*\.?[0-9]+)\s*eV",
        r"Total energy corrected.*?=\s*([-+]?[0-9]*\.?[0-9]+)\s*eV",
    ]
    for pat in patterns:
        m = re.search(pat, out_content)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                continue
    return None


class AimsWorkflowManager:
    def __init__(self, config: Dict):
        self.config = config
        self.ssh = None
        self.current_job_id = None
        self.current_remote_dir = None

    def connect_remote(self) -> Tuple[bool, str]:
        try:
            ssh_config = self.config.get("remote_server", {})
            self.ssh = SSHManager(**ssh_config)
            ok, msg = self.ssh.connect()
            if ok:
                self.ssh.open_sftp()
            return ok, msg
        except Exception as e:
            return False, f"连接失败: {e}"

    def disconnect_remote(self):
        if self.ssh:
            self.ssh.close()

    def generate_slurm_script(
        self,
        job_name: str = "AIMS_Job",
        n_nodes: int = 1,
        n_procs: int = 16,
        time_limit: str = "02:00:00",
        partition: str = "cpu",
        email: Optional[str] = None,
        aims_command: str = "aims.x"
    ) -> str:
        email_directive = f"#SBATCH --mail-user={email}\n#SBATCH --mail-type=END" if email else ""
        return AimsQueueTemplates.get_template("slurm").format(
            job_name=job_name,
            n_nodes=n_nodes,
            n_procs=n_procs,
            time_limit=time_limit,
            partition=partition,
            total_procs=n_nodes * n_procs,
            aims_command=aims_command,
            email_directive=email_directive
        )

    def prepare_calculation(
        self,
        structure_dict: Dict,
        structure_name: str,
        aims_params: Optional[Dict] = None,
        slurm_params: Optional[Dict] = None,
        control_content: Optional[str] = None,
        slurm_content: Optional[str] = None
    ) -> Tuple[bool, Dict]:
        try:
            params = {
                "xc": "pbe",
                "fmax": 0.02,
                "charge": 0,
                "acc_rho": 1e-4,
                "acc_eev": 1e-3,
                "acc_etot": 1e-6,
                "sc_iter": 200,
                "kgrid": [4, 4, 1],
            }
            if aims_params:
                params.update(aims_params)
            k1, k2, k3 = params.get("kgrid", [4, 4, 1])

            if control_content:
                control_str = control_content
            else:
                tmpl = AimsTemplates.get_template("geometry_opt")
                control_str = tmpl.format(
                    xc=params.get("xc"),
                    fmax=params.get("fmax"),
                    charge=params.get("charge"),
                    acc_rho=params.get("acc_rho"),
                    acc_eev=params.get("acc_eev"),
                    acc_etot=params.get("acc_etot"),
                    sc_iter=params.get("sc_iter"),
                    k1=int(k1), k2=int(k2), k3=int(k3)
                )

            geometry_str = _structure_to_geometry_in(structure_dict)

            slurm_defaults = {
                "job_name": f"AIMS_{structure_name}",
                "n_nodes": 1,
                "n_procs": 16,
                "time_limit": "02:00:00",
                "partition": "cpu",
                "aims_command": "aims.x"
            }
            if slurm_params:
                slurm_defaults.update(slurm_params)

            if slurm_content:
                slurm_script = slurm_content
            else:
                slurm_script = self.generate_slurm_script(**slurm_defaults)

            return True, {
                "geometry.in": geometry_str,
                "control.in": control_str,
                "slurm": slurm_script,
                "params": {"structure_name": structure_name, "aims_params": params, "slurm_params": slurm_defaults}
            }
        except Exception as e:
            return False, {"error": str(e)}

    def submit_calculation(self, calculation_files: Dict, use_default_basis: bool = False, basis_path: Optional[str] = None) -> Tuple[bool, str]:
        if not self.ssh:
            return False, "未连接到远程服务器"
        try:
            structure_name = calculation_files["params"]["structure_name"]
            remote_base = self.config.get("remote_paths", {}).get("aims_base", "AIMS_JOBS")
            ts = int(time.time())
            self.current_remote_dir = f"{remote_base}/AIMS_{structure_name}_{ts}"
            ok, msg = self.ssh.mkdir_remote(self.current_remote_dir)
            if not ok:
                return False, msg

            for fname, content in calculation_files.items():
                if fname == "params":
                    continue
                remote_path = f"{self.current_remote_dir}/{fname if fname != 'slurm' else 'slurm.sh'}"
                ok, m = self.ssh.write_remote_file(remote_path, content)
                if not ok:
                    return False, m

            if use_default_basis and basis_path:
                code, out, err = self.ssh.exec_command(f"cp {basis_path} {self.current_remote_dir}/basis.custom" )
                if code != 0:
                    return False, f"复制基组失败: {err}"

            code, out, err = self.ssh.exec_command(f"cd {self.current_remote_dir} && chmod +x slurm.sh && sbatch slurm.sh")
            if code == 0 and "Submitted batch job" in out:
                job_id = out.split()[-1]
                self.current_job_id = job_id
                return True, job_id
            return False, f"提交失败: {out} {err}"
        except Exception as e:
            return False, str(e)

    def poll_and_get_results(self, job_id: Optional[str] = None, check_interval: int = 5, max_wait_seconds: int = 1800, on_poll_callback=None) -> Tuple[bool, Dict]:
        if not self.ssh:
            return False, {"error": "未连接到远程服务器"}
        target_job_id = job_id or self.current_job_id
        if not target_job_id:
            return False, {"error": "未指定Job ID"}
        ok, msg = self.ssh.poll_job_completion(target_job_id, check_interval, max_wait_seconds, on_poll_callback)
        if not ok:
            return False, {"error": msg}
        return self.get_calculation_results(self.current_remote_dir, target_job_id)

    def get_calculation_results(self, remote_dir: str, job_id: Optional[str] = None) -> Tuple[bool, Dict]:
        if not self.ssh:
            return False, {"error": "未连接到远程服务器"}
        try:
            results = {"remote_dir": remote_dir, "job_id": job_id}
            success, out_content = self.ssh.read_remote_file(f"{remote_dir}/aims.out", max_bytes=12000)
            if success:
                results["aims_out"] = out_content
                e = _parse_aims_energy(out_content)
                if e is not None:
                    results["energy"] = e
            success, log_content = self.ssh.read_remote_file(f"{remote_dir}/job.log", max_bytes=4000)
            if success:
                results["log"] = log_content
            results["fetched_at"] = datetime.datetime.now().isoformat()
            return True, results
        except Exception as e:
            return False, {"error": str(e)}

    def download_output(self, local_dir: str, files_to_download: Optional[List[str]] = None) -> Tuple[bool, str]:
        if not self.ssh:
            return False, "未连接到远程服务器"
        if not self.current_remote_dir:
            return False, "无法确定远程工作目录"
        if files_to_download is None:
            files_to_download = ["aims.out", "geometry.in", "geometry.in.next_step"]
        try:
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            for fname in files_to_download:
                remote_file = f"{self.current_remote_dir}/{fname}"
                local_file = str(Path(local_dir) / fname)
                ok, msg = self.ssh.download_result_file(remote_file, local_file)
                if not ok:
                    return False, f"下载 {fname} 失败: {msg}"
            return True, f"下载完成到 {local_dir}"
        except Exception as e:
            return False, str(e)

    def cleanup_remote(self, remote_dir: Optional[str] = None) -> Tuple[bool, str]:
        if not self.ssh:
            return False, "未连接到远程服务器"
        target_dir = remote_dir or self.current_remote_dir
        if not target_dir:
            return False, "无法确定远程工作目录"
        return self.ssh.cleanup_remote_dir(target_dir)

    def fetch_final_structure(self, remote_dir: str) -> Tuple[bool, Optional[Structure]]:
        if not self.ssh:
            return False, None
        candidates = ["geometry.in.next_step", "geometry.in"]
        for fname in candidates:
            ok, content = self.ssh.read_remote_file(f"{remote_dir}/{fname}")
            if ok:
                try:
                    atoms = read(io.StringIO(content), format="aims")
                    s = AseAtomsAdaptor.get_structure(atoms)
                    return True, s
                except Exception:
                    continue
        return False, None
