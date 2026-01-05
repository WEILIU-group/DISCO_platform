import base64
import os
import io
import time
import re
import pandas as pd
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, ctx, no_update, ALL, MATCH
from dash.exceptions import PreventUpdate
from pymatgen.core import Structure
import traceback
import sys
from pathlib import Path
import json

import crystal_toolkit.components as ctc
from crystal_toolkit.settings import SETTINGS

# 路径设置 (用于寻找 utils/SISSO_extract.py)
# 假设当前脚本在 services/sisso/ 或类似子目录下，PROJECT_ROOT 指向项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# 1. SSH 管理类 (真实逻辑)
# =============================================================================
from services.remote_server.ssh_manager import SSHManager as BaseSSHManager


class RealSSHManager:
    """Thin wrapper to reuse shared SSHManager while keeping existing interface."""

    def __init__(self, hostname, username, password, port=22, **kwargs):
        self._base = BaseSSHManager(hostname=hostname, port=int(port), username=username, password=password)

    def connect(self):
        ok, msg = self._base.connect()
        if ok:
            self._base.open_sftp()
        return ok, msg

    def mkdir_remote(self, dir_name):
        return self._base.mkdir_remote(dir_name)

    def write_remote_file(self, filename, content):
        if not isinstance(content, str):
            content = str(content)
        return self._base.write_remote_file(filename, content)

    def exec_command(self, command):
        ret, out, err = self._base.exec_command(command)
        return out, err

    def submit_job_slurm(self, dir_name):
        return self._base.submit_job_slurm(dir_name)

    def check_job_status(self, job_id):
        exists, _ = self._base.query_slurm_status(job_id)
        return "RUNNING" if exists else "COMPLETED"

    def list_remote_files(self, remote_dir):
        return self._base.list_remote_files(remote_dir)

    @property
    def sftp(self):
        return self._base.sftp

    def download_file(self, remote_path, local_path=None):
        success, content = self._base.read_remote_file(remote_path)
        if local_path and success:
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True
        return content if success else None

    def close(self):
        self._base.close()

# =============================================================================
# 2. 核心配置与 DataBuilder
# =============================================================================

CONFIG = {}
DEFAULT_CFG_PATH = Path(__file__).resolve().parent.parent / "services" / "config" / "default_config.json"
try:
    with open(DEFAULT_CFG_PATH, 'r', encoding='utf-8') as f:
        CONFIG = json.load(f)
except Exception:
    CONFIG = {
        "remote_server": {
            "hostname": "127.0.0.1", 
            "username": "user",          
            "password": "password",      
            "port": 22
        },
        "sisso_defaults": {"desc_dim": 2, "fcomplexity": 3}
    }

SSHManager = RealSSHManager

class SissoTrainDataBuilder:
    def __init__(self, df): 
        self.df = df
        
    def build_train_dat(self, structs, targets, indices, feats, parser): 
        lines = []
        prop_name = "Property"
        
        # 特征列表：如果外部未传入，使用默认值
        feature_list = feats if feats and len(feats) > 0 else ["Radius", "Electronegativity"]
        
        # 1. 生成表头 (Header)
        # 逻辑：遍历所有选定的原子索引 -> 遍历所有特征
        header_cols = [prop_name] 
        for idx in indices:
            for feat in feature_list:
                header_cols.append(f"Atom{idx}_{feat}")
                
        lines.append(" ".join(header_cols))
        
        # 2. 生成数据行
        for s in structs:
            # [关键修复] 去除后缀 (.cif, .vasp) 以便匹配 CSV 中的 Key
            clean_name = os.path.splitext(s['filename'])[0]
            
            # 从 CSV 字典获取目标值
            val = targets.get(clean_name, 0.0)
            
            row = [str(val)] 
            
            # 生成模拟特征值
            # 确保生成的列数与 header_cols 长度一致 (减去Property列)
            base_seed = abs(hash(clean_name)) % 100
            total_cols = len(indices) * len(feature_list)
            
            for i in range(total_cols):
                mock_val = (base_seed / 10.0) + (i * 1.5)
                row.append(f"{mock_val:.3f}")
                
            lines.append(" ".join(row))
            
        return "\n".join(lines), len(structs), []

class SissoConfigManager: 
    def __init__(self, filepath): 
        self.filepath = filepath
        self.default_template = """ptype=1
ntask=1
scmt=.false.
desc_dim=2
nsample=298
restart=0
fstore=1
nsf= 15
ops='(+)(-)(*)(/)(^2)(sqrt)'
fcomplexity=5
fmax_min=1e-3
fmax_max=1e5
nf_sis=50000
method_so= 'L0'
fit_intercept=.false.
metric= 'RMSE'
nmodel=400
isconvex=(1,1,...)
bwidth=0.001
"""
        self.raw_content = self.default_template
        if self.filepath and os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    self.raw_content = f.read()
            except Exception as e:
                print(f"[Debug] 读取失败: {e}")

    def update_template(self, params):
        content = self.raw_content
        if 'nsample' in params:
            content = re.sub(r"(nsample\s*=\s*)\d+", f"\\g<1>{params['nsample']}", content)
        if 'nsf' in params:
            content = re.sub(r"(nsf\s*=\s*)\d+", f"\\g<1>{params['nsf']}", content)
        if 'desc_dim' in params:
            content = re.sub(r"(desc_dim\s*=\s*)\d+", f"\\g<1>{params['desc_dim']}", content)
        if 'fcomplexity' in params:
            content = re.sub(r"(fcomplexity\s*=\s*)\d+", f"\\g<1>{params['fcomplexity']}", content)
        if 'ops' in params:
            content = re.sub(r"(ops\s*=\s*)'.*?'", f"ops='{params['ops']}'", content)
        return content

MAX_BATCHES = 12
ELEMENTS_DF = pd.DataFrame()
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths = [os.path.join(base_dir, "elements_properties_all.csv")]
    found_path = next((p for p in search_paths if os.path.exists(p)), None)
    if found_path: ELEMENTS_DF = pd.read_csv(found_path)
except: pass

# =============================================================================
# 3. UI 初始化
# =============================================================================
app = dash.Dash(__name__, assets_folder=SETTINGS.ASSETS_PATH, external_stylesheets=[dbc.themes.BOOTSTRAP, "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css"])
server = app.server

# 工具函数
def parse_structure(content_string: str, fmt: str = None) -> Structure:
    try:
        decoded = base64.b64decode(content_string)
        str_content = decoded.decode("utf-8")
        if fmt is None: fmt = "cif" if ("data_" in str_content[:500] or "_cell_" in str_content[:1000]) else "poscar"
        return Structure.from_str(str_content, fmt=fmt)
    except: return None

def parse_csv_content(content_string):
    if not content_string: return None, 0
    try:
        decoded = base64.b64decode(content_string.split(",")[1])
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.strip()
        df.set_index(df.columns[0], inplace=True)
        return df, len(df)
    except: return None, 0

# UI 组件
new_batch_uploader = dcc.Upload(
    id="new-batch-uploader",
    children=html.Div([
        html.I(className="bi bi-cloud-upload", style={"fontSize": "2rem"}),
        html.Div("拖入文件 (.cif/.vasp)"),
        html.Div("生成新批次", className="text-muted small")
    ]),
    className="upload-container",
    multiple=True
)

task_control_card = dbc.Card([
    dbc.CardHeader("3. 任务控制", className="bg-dark text-white py-2"),
    dbc.CardBody([
        dbc.Button("生成并合并所有批次", id="btn-generate", color="primary", className="w-100 mb-2"),
        dbc.Button("预览文件 / 提交", id="btn-open-editor", outline=True, color="info", className="w-100"),
        dbc.Button("拉取状态", id="btn-pull-status", outline=True, color="warning", className="w-100 mt-2"),
        html.Hr(className="my-2"),
        html.Div(id="log-gen", style={"height": "80px", "overflowY": "scroll", "backgroundColor": "#111", "color": "#0f0", "fontSize": "0.7rem", "whiteSpace": "pre-wrap", "padding": "5px"}),
        html.Div(id="log-sub", style={"height": "60px", "overflowY": "scroll", "backgroundColor": "#222", "color": "#0ff", "fontSize": "0.7rem", "whiteSpace": "pre-wrap", "padding": "5px"})
    ])
])

sisso_settings_card = dbc.Card([
    dbc.CardHeader("2. 全局参数"),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                dbc.Label("维度范围 (Min-Max)", className="small"),
                dbc.InputGroup([
                    dbc.Input(id="inp-dim-min", type="number", value=1, min=1, max=5, size="sm"),
                    dbc.InputGroupText("-", style={"padding": "0 5px"}),
                    dbc.Input(id="inp-dim-max", type="number", value=3, min=1, max=5, size="sm"),
                ], size="sm")
            ], width=6),
            dbc.Col([
                dbc.Label("复杂度范围 (Min-Max)", className="small"),
                dbc.InputGroup([
                    dbc.Input(id="inp-cplx-min", type="number", value=2, min=1, max=10, size="sm"),
                    dbc.InputGroupText("-", style={"padding": "0 5px"}),
                    dbc.Input(id="inp-cplx-max", type="number", value=4, min=1, max=10, size="sm"),
                ], size="sm")
            ], width=6)
        ], className="mb-2"),
        dbc.Label("运算符", className="small"),
        dcc.Dropdown(id="inp-ops", options=[{'label': o, 'value': f'({o})'} for o in ['+', '-', '*', '/', 'exp', 'log', '^2', 'sqrt', 'sin', 'cos']], value=['(+)', '(-)', '(*)', '(/)'], multi=True, style={"fontSize": "0.8rem"}),
        html.Div([
            html.Hr(className="my-2"),
            dbc.Label("特征属性", className="small"),
            dcc.Dropdown(id="feature-columns", options=[{'label': c, 'value': c} for c in ELEMENTS_DF.columns if c not in ['symbol', 'name', 'description']], multi=True, placeholder="留空默认全选", style={"fontSize": "0.8rem"})
        ], id="feature-selection-container")
    ], style={"overflow": "visible"})
], style={"overflow": "visible", "zIndex": 100})

# 新增：直接上传组件
direct_train_uploader = dcc.Upload(
    id="direct-train-uploader",
    children=html.Div([
        html.I(className="bi bi-file-earmark-code", style={"fontSize": "2rem"}),
        html.Div("拖入 train.dat"),
        html.Div("直接使用现有数据", className="text-muted small")
    ]),
    className="upload-container",
    multiple=False
)

file_editor_modal = dbc.Modal([
    dbc.ModalHeader(dbc.ModalTitle("预览与编辑")),
    dbc.ModalBody(dbc.Tabs([
        dbc.Tab(label="SISSO.in", children=[dcc.Textarea(id="editor-sisso", style={"width": "100%", "height": "400px", "fontFamily": "monospace"})]),
        dbc.Tab(label="train.dat", children=[dcc.Textarea(id="editor-train", style={"width": "100%", "height": "400px", "fontFamily": "monospace", "whiteSpace": "pre", "overflowX": "scroll"})])
    ])),
    dbc.ModalFooter([
        dbc.Button("取消", id="btn-close-modal", className="me-2"),
        dcc.Loading(dbc.Button("提交任务", id="btn-submit-modal", color="primary"), type="circle")
    ])
], id="modal-file-editor", size="xl", backdrop=True, style={"zIndex": 10000}, is_open=False)

left_panel = [
    dbc.Card([
        dbc.CardHeader("1. 新建 (New Batch)", className="bg-primary text-white py-2"),
        dbc.CardBody([
            dbc.RadioItems(
                id="tabs-input-mode",
                className="btn-group w-100 mb-3",
                inputClassName="btn-check",
                labelClassName="btn btn-outline-primary",
                labelCheckedClassName="active",
                options=[
                    {"label": "从结构生成", "value": "tab-struct"},
                    {"label": "直接上传", "value": "tab-direct"},
                ],
                value="tab-struct",
            ),
            html.Div(new_batch_uploader, id="content-tab-struct"),
            html.Div([
                direct_train_uploader,
                html.Div(id="direct-upload-status", className="mt-2 text-success small")
            ], id="content-tab-direct", style={"display": "none"})
        ], className="p-2")
    ], className="mb-3"),
    html.Div(sisso_settings_card, className="mb-3"),
    html.Div(task_control_card, className="mb-3")
]
right_panel = [
    dbc.Card([dbc.CardHeader(["批次工作区 (Workspace)", dbc.Button("🗑️ 清空所有", id="btn-reset-all", color="link", size="sm", className="float-end text-decoration-none text-danger py-0")], className="py-2"), 
              dbc.CardBody([html.Div(id="batches-container", className="row g-2"), html.Div("请在左侧拖入结构文件以开始...", id="empty-placeholder", className="text-center text-muted py-5")], className="p-2")], className="mb-3 h-100")
]

ctc.register_crystal_toolkit(app=app, layout=dbc.Container([
    file_editor_modal, 
    dcc.Store(id='store-batches-data', data=[], storage_type='local'), 
    dcc.Store(id='store-job-info', data={}, storage_type='local'), 
    dcc.Interval(id='interval-job-monitor', interval=10000, n_intervals=0),
    dbc.NavbarSimple(
        brand="🧬 SISSO HPC Workflow",
        color="white", className="mb-3 shadow-sm",
        children=[dbc.NavItem(dbc.NavLink("Reset", href="/", external_link=True))]
    ), 
    dbc.Row([dbc.Col(left_panel, width=12, lg=3), dbc.Col(right_panel, width=12, lg=9)]),
    dbc.Row([dbc.Col(dbc.Card([dbc.CardHeader("4. 计算结果"), dbc.CardBody(html.Div(id='result-display'))], className="mt-3"), width=12)])
], fluid=True, style={"minHeight": "100vh", "backgroundColor": "#f8f9fa"}))

# =============================================================================
# 4. 回调函数
# =============================================================================

@app.callback(
    Output("store-batches-data", "data"), Output("batches-container", "children"), Output("empty-placeholder", "style"), Output("new-batch-uploader", "contents"),
    Input("new-batch-uploader", "contents"), Input("btn-reset-all", "n_clicks"),
    State("new-batch-uploader", "filename"), State("store-batches-data", "data"), State("batches-container", "children")
)
def create_new_batch(contents, n_reset, filenames, current_data, current_children):
    if ctx.triggered_id == "btn-reset-all": return [], [], {"display": "block"}, None
    if not contents: raise PreventUpdate
    if current_data is None: current_data = []
    if current_children is None: current_children = []
    
    new_structures = []
    for c, f in zip(contents, filenames):
        new_structures.append({'filename': f, 'content': c.split(",")[1]})
    
    batch_id = len(current_data)
    current_data.append({"id": batch_id, "structures": new_structures})
    
    init_struct = parse_structure(new_structures[0]['content']) if new_structures else None
    
    card = dbc.Col(dbc.Card([
        dbc.CardHeader([dbc.Row([
            dbc.Col([html.Strong(f"#{batch_id+1}"), html.Span(f"{len(new_structures)}", className="badge bg-secondary ms-1")], width="auto"),
            dbc.Col([dcc.Dropdown(id={'type': 'batch-struct-select', 'index': batch_id}, options=[{'label': s['filename'], 'value': i} for i, s in enumerate(new_structures)], value=0, clearable=False)], width=3),
            dbc.Col([dbc.Input(id={'type': 'batch-indices-input', 'index': batch_id}, placeholder="Index (e.g. 48 52)", size="sm")], width=4),
            dbc.Col([dcc.Upload(id={'type': 'batch-csv-upload', 'index': batch_id}, children=html.Div([html.Div([html.I(className="bi bi-file-earmark-arrow-up"), " CSV"], id={'type': 'batch-csv-label', 'index': batch_id}), html.Div(id={'type': 'batch-csv-status', 'index': batch_id}, className="text-success small fw-bold ms-1")]), style={"border": "1px dashed #6c757d", "height": "31px", "cursor": "pointer", "backgroundColor": "#f8f9fa", "display": "flex", "alignItems": "center", "justifyContent": "center"})], width=3)
        ], className="g-1 align-items-center")]),
        dbc.CardBody([ctc.StructureMoleculeComponent(init_struct, id=f"viewer-batch-{batch_id}", color_scheme="VESTA").layout(size="550px")])
    ], className="shadow-sm border-0 mb-3"), width=12, lg=6, xl=6)
    
    current_children.append(card)
    return current_data, current_children, {"display": "none"}, None

@app.callback([Output(f"viewer-batch-{i}", "data") for i in range(MAX_BATCHES)], Input({'type': 'batch-struct-select', 'index': ALL}, 'value'), State("store-batches-data", "data"))
def update_dynamic_viewers(vals, data):
    outs = [no_update] * MAX_BATCHES
    if not data or not vals: return outs
    for i, idx in enumerate(vals):
        if i < len(data) and idx is not None:
            outs[i] = parse_structure(data[i]['structures'][idx]['content'])
    return outs

@app.callback(Output({'type': 'batch-csv-status', 'index': MATCH}, 'children'), Output({'type': 'batch-csv-label', 'index': MATCH}, 'style'), Input({'type': 'batch-csv-upload', 'index': MATCH}, 'contents'), State({'type': 'batch-csv-upload', 'index': MATCH}, 'filename'))
def update_csv_status(c, f):
    if not c: return "", {"display": "block"}
    df, cnt = parse_csv_content(c)
    return f"✓ {f[:5]}..", {"display": "none"} if cnt > 0 else {"display": "block"}

# --- 新增：Tab 切换与直接上传状态 ---
@app.callback(
    Output("content-tab-struct", "style"), 
    Output("content-tab-direct", "style"),
    Output("feature-selection-container", "style"),
    Input("tabs-input-mode", "value")
)
def switch_tab_content(at):
    if at == "tab-direct":
        return {"display": "none"}, {"display": "block"}, {"display": "none"}
    return {"display": "block"}, {"display": "none"}, {"display": "block"}

@app.callback(Output("direct-upload-status", "children"), Input("direct-train-uploader", "contents"), State("direct-train-uploader", "filename"))
def update_direct_status(c, f):
    if c: return f"已加载: {f}"
    return ""

# --- [核心合并逻辑] 列名标准化 ---
@app.callback(
    Output("log-gen", "children"),
    Output("editor-sisso", "value"),
    Output("editor-train", "value"),
    Input("btn-generate", "n_clicks"),
    State("store-batches-data", "data"),
    State({'type': 'batch-indices-input', 'index': ALL}, 'value'),
    State({'type': 'batch-csv-upload', 'index': ALL}, 'contents'),
    State("inp-dim-min", "value"),
    State("inp-dim-max", "value"),
    State("inp-cplx-min", "value"),
    State("inp-cplx-max", "value"),
    State("inp-ops", "value"),
    State("feature-columns", "value"),
    State("tabs-input-mode", "value"),
    State("direct-train-uploader", "contents")
)
def generate_merge(n, batch_data_list, indices_list, csv_contents_list, dim_min, dim_max, cplx_min, cplx_max, ops, feat_cols, active_tab, direct_content):
    if not n: raise PreventUpdate
    logs = ["开始处理..."]
    
    # 占位符模板生成逻辑
    def get_sisso_template(nsample, nsf):
        try:
            cm = SissoConfigManager("services/sisso/templates/SISSO.in")
            # 使用占位符 {{dim}} 和 {{cplx}}
            return cm.update_template({
                "desc_dim": "{{dim}}",
                "nsample": nsample, 
                "nsf": nsf,
                "fcomplexity": "{{cplx}}",
                "ops": "".join(ops) if ops else ""
            })
        except Exception as e:
            return f"Template Error: {e}"

    # --- 分支 1: 直接上传模式 ---
    if active_tab == "tab-direct":
        if not direct_content:
            return "错误: 请先上传 train.dat 文件", "", ""
        
        try:
            # 解析 train.dat
            content_type, content_string = direct_content.split(',')
            decoded = base64.b64decode(content_string)
            final_train_dat = decoded.decode('utf-8')
            
            # 简单解析以获取 nsample 和 nsf
            lines = final_train_dat.strip().split('\n')
            lines = [l for l in lines if l.strip()]
            
            if len(lines) < 2:
                return "错误: train.dat 内容过短", "", ""
                
            header = lines[0].split()
            real_nsample = len(lines) - 1
            # nsf = 列数 - 2 (第一列通常是 Materials, 第二列是 Property)
            real_nsf = len(header) - 2
            
            logs.append(f"【直接模式】已解析 train.dat: nsample={real_nsample}, nsf={real_nsf}")
            logs.append(f"【参数范围】Dim: {dim_min}-{dim_max}, Cplx: {cplx_min}-{cplx_max}")
            
            sisso_in_content = get_sisso_template(real_nsample, real_nsf)
            return "\n".join(logs), sisso_in_content, final_train_dat
            
        except Exception as e:
            return f"解析 train.dat 失败: {e}", "", ""

    # --- 分支 2: 结构生成模式 ---
    if not batch_data_list: raise PreventUpdate
    
    current_feat_list = feat_cols if feat_cols and len(feat_cols) > 0 else ["Radius", "Electronegativity"]
    
    try:
        builder = SissoTrainDataBuilder(ELEMENTS_DF)
    except:
        builder = SissoTrainDataBuilder(ELEMENTS_DF)
        logs.append("[注意] 使用模拟数据生成器")

    all_dfs = []
    
    for i, batch_data in enumerate(batch_data_list):
        try:
            indices_str = str(indices_list[i]).replace(",", " ").strip()
            if not indices_str:
                logs.append(f"Batch #{i+1} 跳过: 未输入原子索引")
                continue
            indices = [int(x) for x in indices_str.split()]
            
            csv_df, _ = parse_csv_content(csv_contents_list[i])
            if csv_df is None: 
                logs.append(f"Batch #{i+1} 跳过: 未上传 CSV")
                continue
            
            targets_map = csv_df.iloc[:, 0].to_dict()
            valid_structs = [s for s in batch_data['structures'] if os.path.splitext(s['filename'])[0] in targets_map]
            
            if not valid_structs:
                logs.append(f"Batch #{i+1} 警告: 无匹配结构")
                continue
            
            dat, _, _ = builder.build_train_dat(valid_structs, targets_map, indices, current_feat_list, parse_structure)
            df_base = pd.read_csv(io.StringIO(dat), sep='\s+')
            
            standard_names = []
            for idx_order in range(len(indices)): 
                for feat_name in current_feat_list:
                    standard_names.append(f"{idx_order + 1}_{feat_name}")
            
            current_cols = list(df_base.columns)
            if "Property" in current_cols:
                raw_feat_cols = [c for c in current_cols if c != "Property"]
                if len(raw_feat_cols) == len(standard_names):
                    rename_map = dict(zip(raw_feat_cols, standard_names))
                    df_base.rename(columns=rename_map, inplace=True)
            
            extra = csv_df.iloc[:, 1:]
            valid_ids = [os.path.splitext(s['filename'])[0] for s in valid_structs]
            if not extra.empty:
                df_base = pd.concat([df_base, extra.loc[valid_ids].reset_index(drop=True)], axis=1)
            
            df_base.insert(0, "materials", [f"b{i+1}_{mid}" for mid in valid_ids])
            all_dfs.append(df_base)
            
        except Exception as e:
            logs.append(f"Batch #{i+1} 异常: {e}")

    if not all_dfs: return "\n".join(logs), "", ""

    final_df = pd.concat(all_dfs, ignore_index=True)
    if final_df.isnull().values.any():
        final_df.fillna(0, inplace=True)

    real_nsample = len(final_df)
    real_nsf = final_df.shape[1] - 2
    
    logs.append(f"【生成成功】nsample={real_nsample}, nsf={real_nsf}")
    logs.append(f"【参数范围】Dim: {dim_min}-{dim_max}, Cplx: {cplx_min}-{cplx_max}")

    sisso_in_content = get_sisso_template(real_nsample, real_nsf)
    return "\n".join(logs), sisso_in_content, final_df.to_string(index=False)

@app.callback(Output("modal-file-editor", "is_open"), Input("btn-generate", "n_clicks"), Input("btn-open-editor", "n_clicks"), Input("btn-close-modal", "n_clicks"), Input("btn-submit-modal", "n_clicks"), State("modal-file-editor", "is_open"))
def toggle_modal(n1, n2, n3, n4, o):
    if ctx.triggered_id in ["btn-generate", "btn-open-editor"]: return True
    if ctx.triggered_id in ["btn-close-modal", "btn-submit-modal"]: return False
    return o

# --- [作业管理] 提交 -> 监控 -> 提取 -> 显示 ---
@app.callback(
    Output("store-job-info", "data"), Output("log-sub", "children"), Output("result-display", "children"),
    Input("btn-submit-modal", "n_clicks"), Input("interval-job-monitor", "n_intervals"), Input("btn-pull-status", "n_clicks"),
    State("editor-sisso", "value"), State("editor-train", "value"), State("store-job-info", "data"), State("log-sub", "children"),
    State("inp-dim-min", "value"), State("inp-dim-max", "value"), State("inp-cplx-min", "value"), State("inp-cplx-max", "value"),
    prevent_initial_call=True
)
def manage_job(n_submit, n_interval, n_pull, sisso_template, train, job_info, current_log, dim_min, dim_max, cplx_min, cplx_max):
    trigger = ctx.triggered_id
    
    # 提交作业
    if trigger == "btn-submit-modal":
        try:
            ssh = SSHManager(**CONFIG.get("remote_server", {}))
            ok, msg = ssh.connect()
            if not ok: return {}, f"连接失败: {msg}", no_update
            
            # 创建主目录
            main_rd = f"SISSO_Batch_{int(time.time())}"
            ssh.mkdir_remote(main_rd)
            
            # 确保范围有效
            d_min = int(dim_min) if dim_min else 1
            d_max = int(dim_max) if dim_max else d_min
            c_min = int(cplx_min) if cplx_min else 1
            c_max = int(cplx_max) if cplx_max else c_min
            
            submitted_jobs = []
            logs = [f"创建主目录: {main_rd}"]
            
            # 遍历所有组合
            for d in range(d_min, d_max + 1):
                for c in range(c_min, c_max + 1):
                    sub_dir_name = f"d{d}_c{c}"
                    full_remote_path = f"{main_rd}/{sub_dir_name}"
                    
                    # 1. 创建子目录
                    ssh.mkdir_remote(full_remote_path)
                    
                    # 2. 替换模板参数
                    # 注意：这里假设 sisso_template 里有 {{dim}} 和 {{cplx}}
                    # 如果没有（比如用户手动改掉了），replace 不会生效，保持原样
                    current_sisso = sisso_template.replace("{{dim}}", str(d)).replace("{{cplx}}", str(c))
                    
                    # 3. 上传文件
                    ssh.write_remote_file(f"{full_remote_path}/SISSO.in", current_sisso)
                    ssh.write_remote_file(f"{full_remote_path}/train.dat", train)
                    
                    # 4. 复制并提交脚本
                    ssh.exec_command(f"cp ~/slurm.sh ~/{full_remote_path}/")
                    ok_sub, jid = ssh.submit_job_slurm(full_remote_path)
                    
                    if ok_sub:
                        submitted_jobs.append(jid)
                        logs.append(f"  [提交] {sub_dir_name} -> JobID {jid}")
                    else:
                        logs.append(f"  [失败] {sub_dir_name} -> {jid}")
            
            ssh.close()
            
            if submitted_jobs:
                # 策略：只监控最后一个提交的任务 ID
                # 这样当最后一个任务完成时，触发提取逻辑（虽然提取逻辑目前可能只会失败或提取部分）
                last_jid = submitted_jobs[-1]
                return {"remote_dir": main_rd, "job_id": last_jid, "status": "submitted", "all_jobs": submitted_jobs}, "\n".join(logs), no_update
            else:
                return {}, "\n".join(logs) + "\n全部提交失败", no_update
                
        except Exception as e: return {}, f"异常: {e}", no_update

    # 监控与提取
    elif trigger in ["interval-job-monitor", "btn-pull-status"]:
        if not job_info or job_info.get("status") != "submitted":
            if trigger == "btn-pull-status":
                return no_update, "没有正在进行的任务或任务信息已丢失（请重新提交）", no_update
            raise PreventUpdate
        
        ssh = SSHManager(**CONFIG.get("remote_server", {}))
        ok, msg = ssh.connect()
        if not ok: return no_update, f"连接中断: {msg}", no_update
        
        # 监控最后一个任务的状态
        status = ssh.check_job_status(job_info["job_id"])
        
        if status == "RUNNING":
            ssh.close()
            base_log = current_log or ""
            # 避免日志无限增长
            if "运行中" not in base_log[-50:]:
                return no_update, f"{base_log}\n[运行中] Job {job_info['job_id']} (及其他) is running...", no_update
            return no_update, no_update, no_update
            
        elif status == "COMPLETED":
            try:
                # >>> 自动执行提取脚本逻辑 >>>
                remote_dir = job_info['remote_dir']
                
                # --- 步骤 A: 在每个子目录运行 SISSO_extract.py ---
                # 1. 寻找本地 SISSO_extract.py
                extract_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SISSO_extract.py")
                if not os.path.exists(extract_script_path):
                    # 备用路径
                    extract_script_path = os.path.join(PROJECT_ROOT, "utils", "SISSO_extract.py")
                
                if os.path.exists(extract_script_path):
                    with open(extract_script_path, "r", encoding="utf-8") as f:
                        extract_content = f.read()
                    
                    # 2. 上传提取脚本到主目录
                    ssh.write_remote_file(f"{remote_dir}/SISSO_extract.py", extract_content)
                    
                    # 3. 获取所有子目录 (d*_c*)
                    # 使用 find 命令查找子目录
                    cmd_find = f"find ~/{remote_dir} -maxdepth 1 -type d -name 'd*_c*'"
                    out_find, _ = ssh.exec_command(cmd_find)
                    subdirs = [p.strip() for p in out_find.split('\n') if p.strip()]
                    
                    # 4. 遍历子目录并执行提取
                    for subdir_path in subdirs:
                        subdir_name = os.path.basename(subdir_path)
                        # 复制脚本到子目录 -> 执行 -> 生成 results.csv
                        # 注意: 这里的路径处理要小心，subdir_path 是绝对路径或相对路径取决于 find 输出
                        # 假设 find 输出的是绝对路径 /home/user/.../d1_c2
                        
                        cmd_extract = (
                            f"cp ~/{remote_dir}/SISSO_extract.py {subdir_path}/ && "
                            f"cd {subdir_path} && "
                            f"python SISSO_extract.py"
                        )
                        ssh.exec_command(cmd_extract)
                else:
                    # 如果找不到提取脚本，记录警告但继续尝试运行 draw.py (可能用户只想画图?)
                    print(f"[Warning] 本地未找到 {extract_script_path}，跳过子目录提取步骤。")

                # --- 步骤 B: 运行 draw.py 汇总绘图 ---
                # 1. 寻找本地 draw.py 脚本
                draw_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "draw.py")
                if not os.path.exists(draw_script_path):
                    draw_script_path = os.path.join(PROJECT_ROOT, "utils", "draw.py")
                
                if os.path.exists(draw_script_path):
                    with open(draw_script_path, "r", encoding="utf-8") as f:
                        script_content = f.read()
                    
                    # 2. 上传脚本到主目录
                    ssh.write_remote_file(f"{remote_dir}/draw.py", script_content)
                    
                    # 3. 远程执行
                    cmd = f"cd ~/{remote_dir} && python draw.py"
                    ssh.exec_command(cmd)
                    
                    # 4. 下载结果文件
                    csv_content = ssh.download_file(f"{remote_dir}/all_models_rmse_complexity.csv")
                    
                    # 尝试下载图片 (假设是 pareto_frontier.png，如果脚本生成了其他名字，这里需要调整)
                    # 先列出文件确认图片名
                    ok_ls, files = ssh.list_remote_files(remote_dir)
                    img_filename = next((f for f in files if f.endswith(".png")), None)
                    img_base64 = None
                    
                    if img_filename:
                        # 读取二进制图片内容
                        try:
                            with ssh.sftp.file(f"{remote_dir}/{img_filename}", "rb") as f:
                                img_bytes = f.read()
                                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                        except Exception as e:
                            print(f"图片下载失败: {e}")

                    ssh.close()
                    job_info["status"] = "finished"

                    # 5. 结果展示
                    display_children = []
                    
                    # 处理 CSV
                    if csv_content:
                        df_res = pd.read_csv(io.StringIO(csv_content))
                        # 落地到 outputs/discriptor
                        local_root = Path(CONFIG.get("local_paths", {}).get("results_root", "./outputs"))
                        out_dir = local_root / "discriptor"
                        out_dir.mkdir(parents=True, exist_ok=True)
                        df_res.to_csv(out_dir / "all_models_rmse_complexity.csv", index=False)
                        
                        # CSV 下载链接
                        csv_href = "data:text/csv;charset=utf-8," + base64.b64encode(csv_content.encode('utf-8')).decode('utf-8')
                        
                        display_children.append(html.H5("📊 描述符统计结果"))
                        display_children.append(html.A(
                            dbc.Button("📥 下载 CSV 数据", color="success", size="sm", className="mb-2"),
                            href=csv_href,
                            download="all_models_rmse_complexity.csv",
                            target="_blank"
                        ))
                        display_children.append(dbc.Table.from_dataframe(df_res, striped=True, bordered=True, hover=True, size="sm", style={"maxHeight": "300px", "overflowY": "scroll"}))
                        display_children.append(html.Hr())

                    # 处理图片
                    if img_base64:
                        img_src = f"data:image/png;base64,{img_base64}"
                        display_children.append(html.H5("📈 帕累托前沿图"))
                        display_children.append(html.A(
                            dbc.Button("📥 下载图片", color="info", size="sm", className="mb-2"),
                            href=img_src,
                            download=img_filename,
                            target="_blank"
                        ))
                        display_children.append(html.Img(src=img_src, style={"maxWidth": "100%", "border": "1px solid #ddd", "padding": "5px"}))

                    if not display_children:
                        return job_info, "作业完成，但未生成有效结果文件 (CSV/PNG)", no_update
                        
                    return job_info, f"作业完成，结果已保存到 {out_dir}", display_children
                else:
                    ssh.close()
                    return job_info, f"作业完成，但在本地未找到 {draw_script_path}，无法自动提取。", no_update
                # <<< 结束 >>>
                
            except Exception as e:
                ssh.close()
                return job_info, f"提取过程出错: {e}", no_update
        
        ssh.close()
    return no_update, no_update, no_update

if __name__ == "__main__":
    # Dash 入口，固定端口 8050，避免与其他模块冲突
    app.run(debug=True, port=8050)