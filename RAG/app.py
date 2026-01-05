"""Simple launcher page for Cherry Studio within the DISCO platform.

- Provides a small Dash UI with one button to start Cherry Studio locally.
- Default executable path is taken from env `CHERRY_STUDIO_PATH` or the provided Windows path.

Run: `python rag/app.py` (adjust port if needed via `PORT` env).
"""
import os
import sys
import subprocess
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, Input, Output, State, no_update

# Default Cherry Studio executable path (override via CHERRY_STUDIO_PATH)
DEFAULT_EXE = Path(os.getenv("CHERRY_STUDIO_PATH", r"D:\cherry\Cherry Studio\Cherry Studio.exe"))
DEFAULT_PORT = int(os.getenv("PORT", "8053"))

app: Dash = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container(
    [
        html.H2("Cherry Studio Launcher", className="mt-3 mb-2"),
        html.P("点击下方按钮在本机启动 Cherry Studio（需本机已安装）", className="text-muted"),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Input(
                        id="exe-path",
                        type="text",
                        value=str(DEFAULT_EXE),
                        debounce=True,
                        placeholder="Cherry Studio.exe 的路径",
                    ),
                    width=9,
                ),
                dbc.Col(
                    dbc.Button("启动 Cherry Studio", id="btn-launch", color="primary", className="w-100"),
                    width=3,
                ),
            ],
            className="g-2 mb-3",
        ),
        dbc.Alert(id="alert", is_open=False, duration=4000),
        html.Div(
            [
                html.Small("提示：如路径不同，可修改上方输入框或设置环境变量 CHERRY_STUDIO_PATH。"),
            ],
            className="text-muted",
        ),
    ],
    fluid=True,
    style={"maxWidth": "900px"},
)


def launch_exe(path_str: str) -> tuple[bool, str]:
    exe_path = Path(path_str).expanduser()
    if not exe_path.exists():
        return False, f"找不到可执行文件: {exe_path}"

    try:
        creation_flags = 0
        if os.name == "nt":
            # Detach so the Dash process不会阻塞关闭
            creation_flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
        subprocess.Popen([str(exe_path)], creationflags=creation_flags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True, f"已启动 Cherry Studio: {exe_path}"
    except Exception as exc:  # pragma: no cover
        return False, f"启动失败: {exc}"


@app.callback(
    Output("alert", "children"),
    Output("alert", "color"),
    Output("alert", "is_open"),
    Input("btn-launch", "n_clicks"),
    State("exe-path", "value"),
    prevent_initial_call=True,
)
def on_launch(n_clicks, path_str):
    ok, msg = launch_exe(path_str or str(DEFAULT_EXE))
    return msg, ("success" if ok else "danger"), True


if __name__ == "__main__":
    # On Windows, ensure UTF-8 output
    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception:
            pass
    app.run_server(debug=True, port=DEFAULT_PORT)
