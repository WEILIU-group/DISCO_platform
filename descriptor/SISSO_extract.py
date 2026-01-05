import numpy as np
import os
import re
import csv

# ===== 基本路径 =====
MODELS_DIR = "Models"
USPACE_EXPR_FILE = "SIS_subspaces/Uspace.expressions"
SISSO_IN_FILE = "SISSO.in"

# ===== 输出文件（当前目录）=====
OUT_CSV = "models.csv"


# ===== 0. 从 SISSO.in 读取 desc_dim =====
def read_desc_dim_from_sisso_in(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        text = f.read()
    m = re.search(r"desc_dim\s*=\s*(\d+)", text)
    if m:
        return int(m.group(1))
    return None


DESC_DIM = read_desc_dim_from_sisso_in(SISSO_IN_FILE)


# ===== 1. 在 Models 目录下自动找 top*_Dxxx 文件（支持 top****_D002 这种） =====
top_candidates = []  # (full_path, fname, D_dim)

if not os.path.isdir(MODELS_DIR):
    raise RuntimeError(f"找不到目录：{MODELS_DIR}（请确认在 SISSO 任务目录下运行）")

for fname in os.listdir(MODELS_DIR):
    if not fname.startswith("top"):
        continue
    if fname.endswith("_coeff"):
        continue
    if "_D" not in fname:
        continue

    full_path = os.path.join(MODELS_DIR, fname)
    m = re.search(r"_D(\d+)", fname)
    d_dim = int(m.group(1)) if m else None

    top_candidates.append((full_path, fname, d_dim))

if not top_candidates:
    raise RuntimeError(
        f"在 {MODELS_DIR}/ 里没有找到形如 top*_Dxxx 的文件，"
        f"请确认 SISSO 是否正常结束。"
    )

# 如果 SISSO.in 里有 desc_dim，则用它过滤维度
if DESC_DIM is not None:
    filtered = [c for c in top_candidates if (c[2] == DESC_DIM or c[2] is None)]
    if filtered:
        top_candidates = filtered

# 选修改时间最新的那个 top 文件
TOP_FILE, top_fname, top_D = max(top_candidates, key=lambda c: os.path.getmtime(c[0]))

# 如果 desc_dim 还没确定，就用文件名里的 D
if DESC_DIM is None:
    if top_D is None:
        raise RuntimeError("无法从文件名和 SISSO.in 推断 desc_dim。")
    DESC_DIM = top_D

# 对应的系数文件：同名加 _coeff
COEFF_FILE = TOP_FILE + "_coeff"


# ===== 2. 从 SISSO.in 里解析 ops 列表（算符集合） =====
def read_ops_from_sisso_in(path: str):
    if not os.path.exists(path):
        return ['+', '-', '*', '/']  # 退回一个简单默认集合

    with open(path, "r") as f:
        text = f.read()

    m = re.search(r"ops\s*=\s*'(.*?)'", text, flags=re.DOTALL)
    if not m:
        return ['+', '-', '*', '/']

    ops_str = m.group(1)
    ops = re.findall(r"\((.*?)\)", ops_str)
    ops = [op for op in ops if op]
    if not ops:
        return ['+', '-', '*', '/']
    return ops


OPS = read_ops_from_sisso_in(SISSO_IN_FILE)
# 避免 "exp-" 被拆成 "exp" + "-"，按长度从长到短排序
OPS_SORTED = sorted(OPS, key=len, reverse=True)


def count_ops_in_expr(expr: str, ops_sorted) -> int:
    """统计一个表达式里使用的算符次数（基于 ops 列表）"""
    s = expr
    total = 0
    for op in ops_sorted:
        pattern = re.escape(op)
        matches = re.findall(pattern, s)
        c = len(matches)
        if c > 0:
            total += c
            # 把已经统计过的部分替换掉，避免重叠计数（如 exp- vs exp）
            s = re.sub(pattern, " ", s)
    return total


# ===== 3. 读 Uspace.expressions，保证“行号 = Feature_ID” =====
if not os.path.exists(USPACE_EXPR_FILE):
    raise RuntimeError(f"找不到文件：{USPACE_EXPR_FILE}")

exprs = []
with open(USPACE_EXPR_FILE, "r") as f:
    for line in f:
        raw = line.rstrip("\n")
        if "SIS_score" in raw:
            expr = raw.split("SIS_score")[0].strip()
        else:
            expr = raw.strip()
        exprs.append(expr)

N_EXPR = len(exprs)


# ===== 4. 读 top 文件，提取每个模型的 Feature_ID（只保留 1..N_EXPR 范围内的） =====
raw_feature_rows = []  # 每行的整数列表

with open(TOP_FILE, "r") as f:
    for line in f:
        s = line.strip()
        if not s:
            continue
        # 跳过明显表头
        if any(key in s for key in ["Rank", "F_ID", "Feature", "RMSE", "MAE", "Error", "SISSO"]):
            continue

        parts = s.split()
        ints_in_line = []
        for p in parts:
            p_clean = p.strip(",")
            if p_clean.lstrip("+-").isdigit():
                ints_in_line.append(int(p_clean))

        if len(ints_in_line) == 0:
            continue

        raw_feature_rows.append(ints_in_line)

if not raw_feature_rows:
    raise RuntimeError(f"在 {TOP_FILE} 里没有读到任何整数行，请检查文件内容。")

feature_ids_list = []
keep_row_idx = []  # 记录哪些行是“合法模型”，方便对齐 coeff

for row_idx, ints in enumerate(raw_feature_rows):
    # 常见情况：每行是 [rank, id1, id2, ...]
    if len(ints) >= DESC_DIM + 1:
        candidate = ints[1:1 + DESC_DIM]
    elif len(ints) == DESC_DIM:
        candidate = ints
    else:
        continue

    # 过滤掉越界的 ID
    if not all(1 <= i <= N_EXPR for i in candidate):
        continue

    feature_ids_list.append(candidate)
    keep_row_idx.append(row_idx)

if not feature_ids_list:
    raise RuntimeError(
        f"{TOP_FILE} 里解析到的模型行都包含越界 Feature_ID，"
        f"请检查 top 文件格式。"
    )

feature_ids_all = np.array(feature_ids_list, dtype=int)
nmodels = feature_ids_all.shape[0]


# ===== 5. 读系数文件 top*_Dxxx_coeff（行与 top 文件对应） =====
coeff_raw = None
if os.path.exists(COEFF_FILE):
    coeff_rows = []
    with open(COEFF_FILE, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # 跳过可能表头
            if any(key in s for key in ["c0", "Coeff", "Rank"]):
                continue
            parts = s.split()
            try:
                floats = [float(p) for p in parts]
            except ValueError:
                continue
            if floats:
                coeff_rows.append(floats)

    if coeff_rows:
        coeff_raw_full = np.array(coeff_rows, dtype=float)
        # 用 keep_row_idx 对齐：只取与 feature_ids_list 保留的行
        coeff_raw = coeff_raw_full[keep_row_idx, :]
    else:
        coeff_raw = None
else:
    coeff_raw = None

has_coeff = coeff_raw is not None


# ===== 6. 读 Uspace 数据：优先找 Uspace_t001.dat，其次 Uspace.dat =====
data = None
for candidate in ["SIS_subspaces/Uspace_t001.dat", "SIS_subspaces/Uspace.dat"]:
    if os.path.exists(candidate):
        try:
            data = np.loadtxt(candidate)
            break
        except Exception:
            pass


def rmse(y, y_pred):
    return np.sqrt(np.mean((y - y_pred) ** 2))


def maxae(y, y_pred):
    return np.max(np.abs(y - y_pred))


# ===== 7. 准备误差相关信息 =====
has_error = False
if has_coeff and (data is not None) and coeff_raw.shape[1] >= DESC_DIM + 1:
    has_error = True
    # 回归：第 1 列是 y_true，后面每列是一个 feature
    y_true = data[:, 0]
    X_feat = data[:, 1:]

    c0_all = coeff_raw[:, 0]
    c_all = coeff_raw[:, 1:1 + DESC_DIM]
    n_iter = min(nmodels, coeff_raw.shape[0])
else:
    if has_coeff and coeff_raw.shape[1] >= DESC_DIM + 1:
        c0_all = coeff_raw[:, 0]
        c_all = coeff_raw[:, 1:1 + DESC_DIM]
        n_iter = min(nmodels, coeff_raw.shape[0])
    else:
        n_iter = nmodels


# ===== 8. 写入当前目录 models.csv =====
rows_written = 0
with open(OUT_CSV, "w", newline="") as fout:
    writer = csv.writer(fout)

    # 表头：rank, expr_1..expr_D, C_comp_1..C_comp_D, C_total, [c0..cD], [RMSE, MaxAE]
    header = ["rank"]
    header += [f"expr_{i+1}" for i in range(DESC_DIM)]
    header += [f"C_comp_{i+1}" for i in range(DESC_DIM)]
    header.append("C_total")
    if has_coeff and coeff_raw.shape[1] >= DESC_DIM + 1:
        header.append("c0")
        header += [f"c{i+1}" for i in range(DESC_DIM)]
    if has_error:
        header += ["RMSE", "MaxAE"]

    writer.writerow(header)
    rows_written += 1

    # 主循环：每一行一个模型
    for rank in range(n_iter):
        ids = feature_ids_all[rank]  # 例如 [123, 456] / [123,456,789]
        expr_list = [exprs[i - 1] for i in ids]

        # 计算每个分量的复杂度和总复杂度
        comp_complexities = [count_ops_in_expr(e, OPS_SORTED) for e in expr_list]
        total_complexity = sum(comp_complexities)

        row = [rank + 1]
        row += expr_list
        row += comp_complexities
        row.append(total_complexity)

        if has_coeff and coeff_raw.shape[1] >= DESC_DIM + 1:
            c0 = c0_all[rank]
            cs = c_all[rank]

            row.append(float(f"{c0:.6f}"))
            for cj in cs:
                row.append(float(f"{cj:.6f}"))

            if has_error:
                cols = ids - 1
                Dmat = X_feat[:, cols]  # (nsample, DESC_DIM)
                y_pred = c0 + np.dot(Dmat, cs)
                e_rmse = rmse(y_true, y_pred)
                e_maxae = maxae(y_true, y_pred)

                row.append(float(f"{e_rmse:.6f}"))
                row.append(float(f"{e_maxae:.6f}"))

        writer.writerow(row)
        rows_written += 1

print("[DONE]")
print("  wrote:", os.path.abspath(OUT_CSV))
print("  top file used:", TOP_FILE)
print("  desc_dim:", DESC_DIM)
print("  models written (excluding header):", rows_written - 1)
print("  has_coeff:", has_coeff, "has_error:", has_error)