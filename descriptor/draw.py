#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
在总目录运行：
  python plot_rmse_complexity_pareto.py

功能：
1) 遍历所有子目录 */models.csv
2) 计算 complexity = sum(C_comp_*) + 1（自动兼容 1D/2D/3D...）
3) 过滤掉 RMSE > 50 的模型
4) 绘制 complexity-RMSE 散点图，并标注全局 Pareto 前沿（同时最小化 complexity 和 RMSE）
5) 导出汇总 CSV 和 Pareto CSV

输出：
- all_models_rmse_complexity.csv
- pareto_front.csv
- rmse_complexity_scatter_pareto.png
"""

from pathlib import Path
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # 无GUI服务器环境
import matplotlib.pyplot as plt


RMSE_MAX = 50.0  # 过滤阈值：RMSE > 50 删除


def load_all_models(root: Path) -> pd.DataFrame:
    rows = []

    for csv_path in sorted(root.glob("*/models.csv")):
        folder = csv_path.parent.name
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[WARN] skip {csv_path} (read failed): {e}")
            continue

        if "RMSE" not in df.columns:
            print(f"[WARN] skip {csv_path} (no RMSE column)")
            continue

        comp_cols = [c for c in df.columns if c.startswith("C_comp_")]
        if not comp_cols:
            print(f"[WARN] skip {csv_path} (no C_comp_* columns)")
            continue

        # 转数值
        df["RMSE"] = pd.to_numeric(df["RMSE"], errors="coerce")
        for c in comp_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # complexity = sum(C_comp_*) + 1
        df["complexity"] = df[comp_cols].sum(axis=1, skipna=True) + 1

        # 拼 descriptor（可选）
        expr_cols = sorted(
            [c for c in df.columns if c.startswith("expr_")],
            key=lambda x: int(x.split("_")[1]),
        )
        if expr_cols:
            df["descriptor"] = df[expr_cols].astype(str).agg(" | ".join, axis=1)
            df["desc_dim"] = len(expr_cols)
        else:
            df["descriptor"] = ""
            df["desc_dim"] = 0

        # rank（可选）
        if "rank" in df.columns:
            df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
        else:
            df["rank"] = pd.NA

        df["folder"] = folder

        keep = df[["folder", "rank", "desc_dim", "complexity", "RMSE", "descriptor"]].copy()
        keep = keep.dropna(subset=["complexity", "RMSE"])
        rows.append(keep)

    if not rows:
        raise RuntimeError("No usable */models.csv found (or missing RMSE/C_comp_* columns).")

    return pd.concat(rows, ignore_index=True)


def pareto_front_min(df: pd.DataFrame, xcol: str, ycol: str) -> pd.DataFrame:
    """
    计算二维 Pareto 前沿（第一前沿）：同时最小化 x 和 y
    算法：按 x 升序（同 x 按 y 升序）排序，从左到右扫描维护 best_y
    """
    d = df.sort_values([xcol, ycol], ascending=[True, True]).reset_index(drop=True)

    best_y = float("inf")
    keep_idx = []
    for i, r in d.iterrows():
        y = float(r[ycol])
        if y < best_y:  # 严格改进才进前沿
            keep_idx.append(i)
            best_y = y

    return d.loc[keep_idx].copy()


def main():
    root = Path(".").resolve()
    all_df = load_all_models(root)

    # 过滤：RMSE <= 50
    all_df = all_df[all_df["RMSE"] <= RMSE_MAX].copy()

    if all_df.empty:
        raise RuntimeError(f"After filtering RMSE <= {RMSE_MAX}, no data left.")

    # 导出汇总
    all_df.to_csv("all_models_rmse_complexity.csv", index=False)

    # Pareto
    pf = pareto_front_min(all_df, "complexity", "RMSE")
    pf.to_csv("pareto_front.csv", index=False)

    # 画图
    plt.figure()
    plt.scatter(all_df["complexity"], all_df["RMSE"], s=10, alpha=0.35, label=f"Models (RMSE <= {RMSE_MAX})")
    plt.scatter(pf["complexity"], pf["RMSE"], s=45, marker="x", label="Pareto front")
    plt.plot(pf["complexity"], pf["RMSE"], linewidth=1)

    plt.xlabel("Model complexity = sum(C_comp_*) + 1")
    plt.ylabel("RMSE")
    plt.title("RMSE vs Complexity (All subfolders)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("rmse_complexity_scatter_pareto.png", dpi=300)

    print("[DONE] Wrote:")
    print("  - all_models_rmse_complexity.csv")
    print("  - pareto_front.csv")
    print("  - rmse_complexity_scatter_pareto.png")
    print(f"[INFO] Filter: RMSE <= {RMSE_MAX}")


if __name__ == "__main__":
    main()

