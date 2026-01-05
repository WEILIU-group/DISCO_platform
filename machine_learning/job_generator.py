import os
import json
from pathlib import Path
from services.common.templates import load_slurm_template
from services.config.loader import load_config, get_queue_defaults
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
ML_TEMPLATE_DIR = ROOT_DIR / "services" / "machine_learning"
LOCAL_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _read_template(name: str) -> str:
    """Read a template from the ML services dir, falling back to legacy local templates."""
    for path in (ML_TEMPLATE_DIR / name, LOCAL_TEMPLATE_DIR / name):
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    raise FileNotFoundError(f"Template {name} not found in {ML_TEMPLATE_DIR} or {LOCAL_TEMPLATE_DIR}")

class JobGenerator:
    """
    Generates SLURM scripts and job files for remote execution.
    """
    
    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Load templates
        self.train_script_template = _read_template("train_script.py")

        # Prefer shared SLURM template (ML-specific first), fallback to local
        try:
            self.slurm_template = load_slurm_template(path=ML_TEMPLATE_DIR / "slurm.sh")
        except FileNotFoundError:
            self.slurm_template = _read_template("slurm.sh")

        cfg = load_config()
        self.queue_defaults = get_queue_defaults(cfg)

    def generate_job_files(
        self,
        job_name: str,
        model_name: str,
        params: dict,
        nodes: int = 1,
        ntasks: int = 1,
        time: str = "24:00:00",
        partition: str = "gpu",
        env_name: str = "ml_env"
    ) -> dict:
        """
        Generate all necessary files for a training job.
        Returns a dictionary of filenames and their content.
        """
        # 1. Config file
        config_content = json.dumps({
            "model_name": model_name,
            "params": params
        }, indent=4)
        
        # 2. Train script (using template)
        # No formatting needed for now as it reads from config.json
        train_script_content = self.train_script_template
        
        # 3. SLURM script
        slurm_content = self.slurm_template.format(
            job_name=job_name,
            nodes=nodes or self.queue_defaults.get("nodes", 1),
            ntasks=ntasks or self.queue_defaults.get("ntasks_per_node", 1),
            time_limit=time or self.queue_defaults.get("time_limit", "24:00:00"),
            partition=partition or self.queue_defaults.get("partition", "gpu"),
            email_directive="",
            command="python train_script.py"
        )
        
        return {
            "config.json": config_content,
            "train_script.py": train_script_content,
            "slurm.sh": slurm_content
        }

    def save_job_files(self, files: dict):
        """
        Save generated files to the output directory.
        """
        for filename, content in files.items():
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, "w", newline='\n') as f:
                f.write(content)
        return self.output_dir

