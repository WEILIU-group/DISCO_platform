"""
远程服务器管理模块
负责所有 SSH 连接、文件传输、队列监控等通用操作
"""

import time
import datetime
import paramiko
from typing import Tuple, Optional, List, Dict
from io import StringIO


class SSHManager:
    """SSH 连接管理器 - 支持所有计算任务的远程操作"""

    def __init__(self, hostname: str, port: int, username: str, password: str):
        """
        初始化 SSH 配置

        Args:
            hostname: 服务器地址
            port: SSH 端口
            username: 用户名
            password: 密码
        """
        self.config = {
            "hostname": hostname,
            "port": port,
            "username": username,
            "password": password
        }
        self.ssh = None
        self.sftp = None

    def connect(self) -> Tuple[bool, str]:
        """
        建立 SSH 连接

        Returns:
            (成功标志, 信息文本)
        """
        try:
            self.ssh = paramiko.SSHClient()
            self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh.connect(**self.config, timeout=10)
            return True, "连接成功"
        except Exception as e:
            return False, f"连接失败: {str(e)}"

    def open_sftp(self) -> bool:
        """打开 SFTP 连接"""
        try:
            if self.ssh:
                self.sftp = self.ssh.open_sftp()
                return True
            return False
        except:
            return False

    def close(self):
        """关闭所有连接"""
        if self.sftp:
            try:
                self.sftp.close()
            except:
                pass
        if self.ssh:
            try:
                self.ssh.close()
            except:
                pass

    def mkdir_remote(self, path: str) -> Tuple[bool, str]:
        """在远程服务器创建目录 (使用 mkdir -p)"""
        try:
            if not self.ssh:
                return False, "SSH 未连接"
            
            # 使用 mkdir -p 递归创建目录，且如果目录已存在也不会报错
            cmd = f"mkdir -p '{path}'"
            stdin, stdout, stderr = self.ssh.exec_command(cmd)
            exit_status = stdout.channel.recv_exit_status()
            
            if exit_status == 0:
                return True, f"创建目录成功: {path}"
            else:
                err_msg = stderr.read().decode('utf-8').strip()
                return False, f"创建目录失败: {err_msg}"
        except Exception as e:
            return False, f"创建目录异常: {str(e)}"

    def upload_file(self, local_path: str, remote_path: str) -> Tuple[bool, str]:
        """上传本地文件到远程服务器"""
        try:
            if not self.sftp:
                return False, "SFTP 未连接"
            self.sftp.put(local_path, remote_path)
            return True, f"上传成功: {remote_path}"
        except Exception as e:
            return False, f"上传失败: {str(e)}"

    def write_remote_file(self, remote_path: str, content: str) -> Tuple[bool, str]:
        """写入内容到远程文件"""
        try:
            if not self.sftp:
                return False, "SFTP 未连接"
            with self.sftp.file(remote_path, "w") as f:
                f.write(content)
            return True, f"写入文件成功: {remote_path}"
        except Exception as e:
            return False, f"写入文件失败: {str(e)}"

    def read_remote_file(self, remote_path: str, max_bytes: Optional[int] = None) -> Tuple[bool, str]:
        """
        读取远程文件内容

        Args:
            remote_path: 远程文件路径
            max_bytes: 最多读取字节数（从文件末尾）

        Returns:
            (成功标志, 文件内容)
        """
        try:
            if not self.sftp:
                return False, "SFTP 未连接"

            with self.sftp.file(remote_path, "r") as f:
                if max_bytes:
                    f.seek(0, 2)
                    size = f.tell()
                    f.seek(max(0, size - max_bytes))
                content = f.read().decode('utf-8', errors='ignore')
            return True, content
        except Exception as e:
            return False, f"读取文件失败: {str(e)}"

    def list_remote_files(self, remote_dir: str) -> Tuple[bool, List[str]]:
        """列出远程目录文件"""
        try:
            if not self.sftp:
                return False, []
            files = self.sftp.listdir(remote_dir)
            return True, files
        except Exception as e:
            return False, []

    def exec_command(self, cmd: str) -> Tuple[int, str, str]:
        """
        执行远程命令

        Args:
            cmd: 要执行的命令

        Returns:
            (返回码, 标准输出, 标准错误)
        """
        try:
            stdin, stdout, stderr = self.ssh.exec_command(cmd)
            out = stdout.read().decode().strip()
            err = stderr.read().decode().strip()
            ret_code = stdout.channel.recv_exit_status()
            return ret_code, out, err
        except Exception as e:
            return -1, "", str(e)

    def submit_job_slurm(self, remote_dir: str, script_name: str = "slurm.sh") -> Tuple[bool, str]:
        """
        提交 Slurm 任务

        Args:
            remote_dir: 远程工作目录
            script_name: 队列脚本名称

        Returns:
            (成功标志, Job ID 或 错误信息)
        """
        cmd = f"cd {remote_dir} && sbatch {script_name}"
        ret_code, out, err = self.exec_command(cmd)

        if ret_code == 0 and "Submitted batch job" in out:
            job_id = out.split()[-1]
            return True, job_id
        else:
            return False, f"提交失败: {out} {err}"

    def query_slurm_status(self, job_id: str) -> Tuple[bool, Optional[str]]:
        """
        查询 Slurm 任务状态

        Args:
            job_id: 任务 ID

        Returns:
            (存在标志, 状态信息/None)
        """
        ret_code, out, err = self.exec_command(f"squeue -h -j {job_id}")
        if ret_code == 0 and out:
            return True, out
        return False, None

    def poll_job_completion(
        self,
        job_id: str,
        check_interval: int = 5,
        max_wait_seconds: int = 1800,
        on_poll_callback=None
    ) -> Tuple[bool, str]:
        """
        轮询等待任务完成

        Args:
            job_id: 任务 ID
            check_interval: 检查间隔（秒）
            max_wait_seconds: 最大等待时间（秒）
            on_poll_callback: 每次轮询回调函数 (job_id, elapsed_time, logs_list)

        Returns:
            (完成标志, 完成消息)
        """
        logs = []
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time

            if elapsed > max_wait_seconds:
                msg = f"等待超时 ({max_wait_seconds}s)，停止轮询"
                logs.append(msg)
                if on_poll_callback:
                    on_poll_callback(job_id, elapsed, logs)
                return False, msg

            exists, status = self.query_slurm_status(job_id)
            if not exists:
                msg = f"任务已从队列消失，计算完成 (耗时 {elapsed:.1f}s)"
                logs.append(msg)
                if on_poll_callback:
                    on_poll_callback(job_id, elapsed, logs)
                return True, msg

            if on_poll_callback:
                on_poll_callback(job_id, elapsed, logs)

            time.sleep(check_interval)

    def download_result_file(
        self,
        remote_path: str,
        local_path: str
    ) -> Tuple[bool, str]:
        """下载结果文件"""
        try:
            if not self.sftp:
                return False, "SFTP 未连接"
            self.sftp.get(remote_path, local_path)
            return True, f"下载成功: {local_path}"
        except Exception as e:
            return False, f"下载失败: {str(e)}"

    def cleanup_remote_dir(self, remote_dir: str) -> Tuple[bool, str]:
        """删除远程目录及其内容"""
        try:
            ret_code, out, err = self.exec_command(f"rm -rf {remote_dir}")
            if ret_code == 0:
                return True, f"清理成功: {remote_dir}"
            return False, f"清理失败: {err}"
        except Exception as e:
            return False, str(e)
