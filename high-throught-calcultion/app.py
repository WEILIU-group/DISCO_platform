import os
import sys
# 将项目根目录添加到 sys.path，确保能导入 utils 和 config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import zipfile
import math
import re
import base64
import itertools  # 新增：用于笛卡尔积计算
import time
import datetime
import json
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, dash_table, no_update, ctx, ALL
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import matplotlib
matplotlib.use('Agg') # 设置后端为Agg，避免GUI报错
import matplotlib.pyplot as plt
from utils.draw_step import step_graph, NO2RR, NO2RR1, HER, OER, CO2RR, ORR

# --- 科学计算库 ---
try:
    from pymatgen.core import Structure, Lattice, Element
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.core.surface import SlabGenerator
    from pymatgen.analysis.adsorption import AdsorbateSiteFinder
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from mp_api.client import MPRester
    from ase import Atoms
    from ase.build import molecule
    from ase.constraints import FixAtoms
    from ase.optimize import BFGS
except ImportError as e:
    print(f"❌ 缺少必要的科学计算库: {e}")

# --- Crystal Toolkit (可选) ---
HAS_CTC = False
try:
    import crystal_toolkit.components as ctc
    HAS_CTC = True
except ImportError:
    print("⚠️ 未检测到 crystal_toolkit，3D 预览功能将不可用。")

# --- DeepMD (可选) ---
HAS_DEEPMD = False
try:
    from deepmd.calculator import DP
    HAS_DEEPMD = True
except:
    pass

# --- 配置管理 ---
try:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from services.config.manager import ConfigManager
    from services.vasp.template_manager import (
        VaspCalculationTemplates,
        QueueSystemTemplates,
        SESSION_EDITOR
    )
    from services.aims.template_manager import AimsTemplates
    from services.structure_optimization.genetic_algorithm import GeneticAdsorptionSearch
    CONFIG = ConfigManager()
    HAS_CONFIG = True
except Exception as e:
    print(f"⚠️ 配置加载失败: {e}")
    HAS_CONFIG = False
    CONFIG = {}
    SESSION_EDITOR = None

# ================= 全局配置 =================
MODEL_PATH = r"C:\Users\hp\Desktop\SCRIPTS\pre_models\DPA-3.1-3M.pt" 
DEFAULT_MP_KEY = "31HfDNN66lqSNhq4YH6zCxTQ2Re9t6cD"
dft_job_tracker = {}  # Track job_id -> {structure_id, remote_dir, submitted_time}

# ================= 1. 周期表数据 (保持不变) =================
CAT_COLORS = {
    "alkali": "#F4D03F", "alkaline": "#F9E79F", "trans": "#F1948A",
    "post-trans": "#AED6F1", "metalloid": "#76D7C4", "nonmetal": "#82E0AA",
    "noble": "#D7BDE2", "lanthanoid": "#F5CBA7", "actinoid": "#F0B27A",
    "empty": "transparent", "indicator": "#D6EAF8"
}

PT_DATA = [
    [("H", "nonmetal"), "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ("He", "noble")],
    [("Li", "alkali"), ("Be", "alkaline"), "", "", "", "", "", "", "", "", "", "", ("B", "metalloid"), ("C", "nonmetal"), ("N", "nonmetal"), ("O", "nonmetal"), ("F", "nonmetal"), ("Ne", "noble")],
    [("Na", "alkali"), ("Mg", "alkaline"), "", "", "", "", "", "", "", "", "", "", ("Al", "post-trans"), ("Si", "metalloid"), ("P", "nonmetal"), ("S", "nonmetal"), ("Cl", "nonmetal"), ("Ar", "noble")],
    [("K", "alkali"), ("Ca", "alkaline"), ("Sc", "trans"), ("Ti", "trans"), ("V", "trans"), ("Cr", "trans"), ("Mn", "trans"), ("Fe", "trans"), ("Co", "trans"), ("Ni", "trans"), ("Cu", "trans"), ("Zn", "trans"), ("Ga", "post-trans"), ("Ge", "metalloid"), ("As", "metalloid"), ("Se", "nonmetal"), ("Br", "nonmetal"), ("Kr", "noble")],
    [("Rb", "alkali"), ("Sr", "alkaline"), ("Y", "trans"), ("Zr", "trans"), ("Nb", "trans"), ("Mo", "trans"), ("Tc", "trans"), ("Ru", "trans"), ("Rh", "trans"), ("Pd", "trans"), ("Ag", "trans"), ("Cd", "trans"), ("In", "post-trans"), ("Sn", "post-trans"), ("Sb", "metalloid"), ("Te", "metalloid"), ("I", "nonmetal"), ("Xe", "noble")],
    [("Cs", "alkali"), ("Ba", "alkaline"), ("57-71", "indicator"), ("Hf", "trans"), ("Ta", "trans"), ("W", "trans"), ("Re", "trans"), ("Os", "trans"), ("Ir", "trans"), ("Pt", "trans"), ("Au", "trans"), ("Hg", "trans"), ("Tl", "post-trans"), ("Pb", "post-trans"), ("Bi", "post-trans"), ("Po", "metalloid"), ("At", "nonmetal"), ("Rn", "noble")],
    [("Fr", "alkali"), ("Ra", "alkaline"), ("89-103", "indicator"), ("Rf", "trans"), ("Db", "trans"), ("Sg", "trans"), ("Bh", "trans"), ("Hs", "trans"), ("Mt", "trans"), ("Ds", "trans"), ("Rg", "trans"), ("Cn", "trans"), ("Nh", "post-trans"), ("Fl", "post-trans"), ("Mc", "post-trans"), ("Lv", "post-trans"), ("Ts", "nonmetal"), ("Og", "noble")],
]
LANT_ROW = [("La", "lanthanoid"), ("Ce", "lanthanoid"), ("Pr", "lanthanoid"), ("Nd", "lanthanoid"), ("Pm", "lanthanoid"), ("Sm", "lanthanoid"), ("Eu", "lanthanoid"), ("Gd", "lanthanoid"), ("Tb", "lanthanoid"), ("Dy", "lanthanoid"), ("Ho", "lanthanoid"), ("Er", "lanthanoid"), ("Tm", "lanthanoid"), ("Yb", "lanthanoid"), ("Lu", "lanthanoid")]
ACT_ROW = [("Ac", "actinoid"), ("Th", "actinoid"), ("Pa", "actinoid"), ("U", "actinoid"), ("Np", "actinoid"), ("Pu", "actinoid"), ("Am", "actinoid"), ("Cm", "actinoid"), ("Bk", "actinoid"), ("Cf", "actinoid"), ("Es", "actinoid"), ("Fm", "actinoid"), ("Md", "actinoid"), ("No", "actinoid"), ("Lr", "actinoid")]

def create_periodic_table():
    def make_cell(item, width="38px", height="38px", font_size="13px"):
        if not item: return html.Div(style={"width": width, "height": height, "margin": "1px"})
        sym, cat = item
        if "-" in sym:
             return html.Div(sym, style={"width": width, "height": height, "margin": "1px", "display":"flex", "alignItems":"center", "justifyContent":"center", "fontSize":"10px", "color":"#555", "backgroundColor": CAT_COLORS[cat], "border": "1px dashed #aaa"})
        return dbc.Button(sym, id={"type": "pt-btn", "elem": sym}, n_clicks=0, className="pt-element-btn shadow-sm", style={"width": width, "height": height, "margin": "1px", "backgroundColor": CAT_COLORS[cat], "color": "#222", "border": "1px solid rgba(0,0,0,0.1)", "padding": "0", "fontWeight": "bold", "fontSize": font_size})

    rows = []
    for r_data in PT_DATA: rows.append(html.Div([make_cell(x) for x in r_data], style={"display": "flex", "justifyContent": "center"}))
    rows.append(html.Div(style={"height": "20px"}))
    l_cells = [html.Div("Lanthanides", style={"width": "116px", "marginRight":"5px", "textAlign":"right", "fontSize":"10px", "display":"flex", "alignItems":"center", "justifyContent":"flex-end", "color":"#666"})] + [make_cell(x) for x in LANT_ROW]
    rows.append(html.Div(l_cells, style={"display": "flex", "justifyContent": "center"}))
    a_cells = [html.Div("Actinides", style={"width": "116px", "marginRight":"5px", "textAlign":"right", "fontSize":"10px", "display":"flex", "alignItems":"center", "justifyContent":"flex-end", "color":"#666"})] + [make_cell(x) for x in ACT_ROW]
    rows.append(html.Div(a_cells, style={"display": "flex", "justifyContent": "center"}))
    return html.Div(rows, className="p-4", style={"backgroundColor": "white", "borderRadius": "10px", "border": "1px solid #ddd"})

# ================= 2. 逻辑处理 =================
class OCPLogic:
    @staticmethod
    def parse_upload(contents, filename):
        if not contents or not filename: return None, "Empty content"
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string).decode('utf-8')
            try: struct = Structure.from_str(decoded, fmt="cif")
            except: 
                try: struct = Structure.from_str(decoded, fmt="poscar")
                except: return None, f"Unsupported format: {filename}"
            
            clean_name = filename.split('.')[0]
            obj = {
                "struct": struct.as_dict(),
                "meta": {"Formula": struct.composition.reduced_formula, "Info": f"File: {filename}", "Source": "Upload"},
                "type": "bulk"
            }
            return {clean_name: obj}, None
        except Exception as e: return None, f"Parse Error: {str(e)}"

    @staticmethod
    def smart_search_mp(api_key, query_str, limit=50):
        key = api_key if api_key else DEFAULT_MP_KEY
        query_str = query_str.strip()
        filters = {"is_stable": True}
        search_mode = "Formula"
        if not query_str: return {}, "Empty query"

        if query_str.startswith("mp-") or query_str.startswith("mvc-"):
            filters["material_ids"] = [query_str]
            search_mode = "ID"
        elif "-" in query_str and not any(char.isdigit() for char in query_str):
            filters["chemsys"] = query_str
            search_mode = "Chemsys"
        elif "," in query_str or " " in query_str:
            elems = [e.strip() for e in re.split(r'[,\s]+', query_str) if e.strip()]
            filters["elements"] = elems
            search_mode = "Elements"
        else:
            filters["formula"] = query_str

        try:
            with MPRester(key) as mpr:
                docs = mpr.materials.summary.search(**filters, fields=["material_id", "formula_pretty", "symmetry", "structure", "energy_per_atom"])
                if len(docs) > limit: docs = docs[:limit]
                if not docs: return {}, f"No results: {query_str}"
                results = {}
                for d in docs:
                    # Safe attribute access
                    mat_id = str(getattr(d, "material_id", "Unknown"))
                    formula = getattr(d, "formula_pretty", "Unknown")
                    
                    symmetry = getattr(d, "symmetry", None)
                    sys_str = str(symmetry.crystal_system) if symmetry else "Unknown"
                    
                    energy = getattr(d, "energy_per_atom", None)
                    energy_str = f"{energy:.4f} eV/atom" if energy is not None else ""
                    
                    struct = getattr(d, "structure", None)
                    if not struct: continue

                    results[mat_id] = {
                        "struct": struct.as_dict(),
                        "meta": {
                            "Formula": formula, 
                            "System": sys_str, 
                            "Search Mode": search_mode,
                            "Energy": energy_str
                        },
                        "type": "bulk"
                    }
                return results, None
        except Exception as e: 
            import traceback
            traceback.print_exc()
            return {}, f"API Error: {str(e)}"

    @staticmethod
    def generate_slabs(parent_id, parent_struct_dict, h, k, l, min_size):
        try:
            bulk = Structure.from_dict(parent_struct_dict)
            sga = SpacegroupAnalyzer(bulk)
            std = sga.get_conventional_standard_structure()
            slab_gen = SlabGenerator(std, (int(h), int(k), int(l)), 10.0, 15.0, center_slab=True, reorient_lattice=True)
            slabs = slab_gen.get_slabs()
            if not slabs: return {}
            slab = slabs[0]
            a, b = slab.lattice.abc[:2]
            ra, rb = math.ceil(min_size/a), math.ceil(min_size/b)
            if ra > 1 or rb > 1: slab.make_supercell([ra, rb, 1])
            new_id = f"{parent_id}_s_{h}{k}{l}"
            return {new_id: {"struct": slab.as_dict(), "meta": {"Parent": parent_id, "Info": f"Slab ({h}{k}{l})"}, "type": "slab"}}
        except Exception as e: 
            print(f"Slice Error: {e}")
            return {}

    @staticmethod
    def generate_ads(parent_id, parent_struct_dict, mol_name, site_filter='all', target_atom_idx=None):
        """稳健版吸附生成逻辑 (Ontop强制 + 方向判断)"""
        try:
            ase_mol = molecule(mol_name)
            pmg_mol = AseAtomsAdaptor.get_molecule(ase_mol)
            slab = Structure.from_dict(parent_struct_dict)
            final_coords = []
            
            target_atom = None
            if target_atom_idx is not None and str(target_atom_idx).strip() != "":
                try:
                    idx = int(target_atom_idx)
                    if 0 <= idx < len(slab):
                        target_atom = slab[idx]
                except ValueError: pass

            # --- 策略 A: 强制生成 Ontop (手动计算) ---
            if target_atom is not None and site_filter in ['all', 'ontop']:
                z_coords = slab.cart_coords[:, 2]
                z_center = (np.min(z_coords) + np.max(z_coords)) / 2.0
                direction = 1.0 if target_atom.coords[2] >= z_center else -1.0
                ontop_coord = target_atom.coords + np.array([0, 0, 2.0 * direction])
                final_coords.append((ontop_coord, 'ontop_manual'))

            # --- 策略 B: 搜索 Bridge/Hollow (使用 ASF) ---
            if target_atom is None or site_filter in ['all', 'bridge', 'hollow']:
                if site_filter != 'ontop':
                    asf = AdsorbateSiteFinder(slab)
                    sym_reduce = 0 if target_atom is not None else 0.1
                    sites_dict = asf.find_adsorption_sites(distance=2.0, symm_reduce=sym_reduce)
                    search_keys = ['bridge', 'hollow'] if site_filter == 'all' else ([site_filter] if site_filter != 'ontop' else [])
                    
                    for k in search_keys:
                        if k not in sites_dict: continue
                        for coord in sites_dict[k]:
                            if target_atom is None:
                                final_coords.append((coord, k))
                                continue
                            dist = np.linalg.norm(coord - target_atom.coords)
                            if dist < 2.8:
                                final_coords.append((coord, k))

            if not final_coords: return {}

            results = {}
            for i, (coord, site_type) in enumerate(final_coords):
                new_struct = AdsorbateSiteFinder(slab).add_adsorbate(pmg_mol, coord, reorient=True)
                info_str = f"Ads {mol_name} on {site_type}"
                if target_atom_idx is not None: info_str += f" (Idx {target_atom_idx})"
                new_id = f"{parent_id}_{mol_name}_{site_type}_{i}"
                if target_atom_idx: new_id += f"_idx{target_atom_idx}"

                results[new_id] = {
                    "struct": new_struct.as_dict(),
                    "meta": {"Parent": parent_id, "Info": info_str},
                    "type": "ads"
                }
            return results
        except Exception as e:
            print(f"Ads Error: {e}")
            return {}

    @staticmethod
    def search_adsorption(parent_id, parent_struct_dict, mol_name, mode="enum", target_atom_idx=None, ga_params=None, top_k=5):
        """
        按指定策略生成或搜索吸附构型。

        mode:
            - "enum": 旧版枚举位点
            - "ga": 遗传算法搜索
            - "minhop": 简易随机+局部优化搜索，近似替代 minima hopping
        """
        mode = (mode or "enum").lower()

        # 退化到旧逻辑
        if mode == "enum":
            return OCPLogic.generate_ads(parent_id, parent_struct_dict, mol_name, site_filter="all", target_atom_idx=target_atom_idx)

        results = {}
        try:
            slab = Structure.from_dict(parent_struct_dict)
            ga_cfg = ga_params or {}

            if mode == "ga":
                searcher = GeneticAdsorptionSearch(
                    slab,
                    mol_name,
                    population_size=int(ga_cfg.get("pop", 12)),
                    n_generations=int(ga_cfg.get("gen", 10)),
                    mutation_rate=float(ga_cfg.get("mut", 0.2)),
                    crossover_rate=float(ga_cfg.get("cx", 0.7)),
                )
                structs = searcher.search(verbose=False)
            else:
                # 简易“min-hop”：随机采样若干初始构型 + 短 BFGS 优化
                from ase.io import read, write
                from ase.optimize import BFGS
                from pymatgen.io.ase import AseAtomsAdaptor
                import random

                def random_orientation():
                    import numpy as np
                    return {
                        "theta": random.uniform(0, 2*np.pi),
                        "phi": random.uniform(0, np.pi),
                        "psi": random.uniform(0, 2*np.pi),
                        "height": random.uniform(1.5, 3.0),
                        "x": random.uniform(*searcher.x_range),
                        "y": random.uniform(*searcher.y_range)
                    }

                searcher = GeneticAdsorptionSearch(slab, mol_name, population_size=ga_cfg.get("pop", 10), n_generations=1)
                candidates = [random_orientation() for _ in range(int(ga_cfg.get("pop", 10)))]
                structs = []
                for cand in candidates:
                    try:
                        ats = searcher._individual_to_structure(cand)
                        opt = BFGS(ats, logfile=None)
                        opt.run(fmax=0.1, steps=20)
                        e = ats.get_potential_energy()
                        structs.append((AseAtomsAdaptor.get_structure(ats), e))
                    except Exception:
                        continue

            if not structs:
                return {}

            # 取能量最低的前 top_k
            structs = sorted(structs, key=lambda x: x[1])[:max(1, int(top_k))]
            for idx, (pmg_struct, energy) in enumerate(structs):
                new_id = f"{parent_id}_{mol_name}_{mode}_{idx}"
                results[new_id] = {
                    "struct": pmg_struct.as_dict(),
                    "meta": {
                        "Parent": parent_id,
                        "Info": f"{mode.upper()} {mol_name} #{idx}",
                        "Energy": f"{energy:.4f} eV"
                    },
                    "type": "ads"
                }
        except Exception as e:
            print(f"Ads Search Error: {e}")
            return {}

        return results

    # === 新增：反应路径中间体批量吸附 ===
    @staticmethod
    def generate_reaction_intermediates(parent_id, parent_struct_dict, reaction_type, target_atom_idx=None):
        """
        在指定位点批量添加反应中间体
        
        Args:
            parent_id: 父结构ID
            parent_struct_dict: 表面结构字典
            reaction_type: 反应类型 ('N2RR', 'ORR', 'CORR')
            target_atom_idx: 目标原子指标（添加位置）
            
        Returns:
            (结构字典, 提示信息)
        """
        try:
            from ase.build import molecule
            
            slab = Structure.from_dict(parent_struct_dict)
            
            # 定义各反应的中间体
            reaction_pathways = {
                'N2RR': [
                    ('N2', '氮气'),      # N≡N
                    ('N', '单N原子'),    # N
                ],
                'ORR': [
                    ('O2', '氧气'),
                    ('OH', '羟基'),
                    ('O', '单O原子'),
                ],
                'CORR': [
                    ('CO2', '二氧化碳'),
                    ('CO', '一氧化碳'),
                    ('C', '单C原子'),
                ]
            }
            
            if reaction_type not in reaction_pathways:
                return {}, f"未知反应类型: {reaction_type}。支持: {list(reaction_pathways.keys())}"
            
            intermediates = reaction_pathways[reaction_type]
            results = {}
            
            # 验证目标原子
            target_atom = None
            if target_atom_idx is not None:
                try:
                    idx = int(target_atom_idx)
                    if 0 <= idx < len(slab):
                        target_atom = slab[idx]
                    else:
                        return {}, f"原子索引 {idx} 超出范围 (0-{len(slab)-1})"
                except ValueError:
                    return {}, f"无效的原子索引: {target_atom_idx}"
            else:
                return {}, "必须指定目标原子索引"
            
            # 批量添加中间体
            z_coords = slab.cart_coords[:, 2]
            z_center = (np.min(z_coords) + np.max(z_coords)) / 2.0
            direction = 1.0 if target_atom.coords[2] >= z_center else -1.0
            
            for mol_name, mol_desc in intermediates:
                try:
                    # 获取分子
                    ase_mol = molecule(mol_name)
                    pmg_mol = AseAtomsAdaptor.get_molecule(ase_mol)
                    
                    # 计算吸附位置 (Ontop)
                    ontop_coord = target_atom.coords + np.array([0, 0, 2.0 * direction])
                    
                    # 添加吸附物
                    new_struct = AdsorbateSiteFinder(slab).add_adsorbate(pmg_mol, ontop_coord, reorient=True)
                    
                    # 生成ID和元数据
                    new_id = f"{parent_id}_{reaction_type}_{mol_name}_idx{target_atom_idx}"
                    
                    results[new_id] = {
                        "struct": new_struct.as_dict(),
                        "meta": {
                            "Parent": parent_id,
                            "Reaction": reaction_type,
                            "Info": f"{reaction_type} 中间体: {mol_desc}",
                            "Molecule": mol_name,
                            "SiteIdx": target_atom_idx
                        },
                        "type": "ads_rxn"  # 标记为反应中间体
                    }
                    
                except Exception as e:
                    print(f"添加 {mol_name} 失败: {e}")
                    continue
            
            if not results:
                return {}, f"无法添加任何 {reaction_type} 中间体"
            
            return results, f"成功添加 {len(results)} 个 {reaction_type} 中间体"
            
        except Exception as e:
            return {}, f"错误: {str(e)}"

    # === 新增：高通量替换功能 ===
    @staticmethod
    def generate_substitutions(parent_id, parent_struct_dict, sub_rule_str):
        """
        根据规则进行原子替换。
        输入格式: "0:Co,Ni; 4:Fe"
        """
        try:
            struct = Structure.from_dict(parent_struct_dict)
            results = {}
            
            # 1. 解析规则字符串
            # 目标结构：sub_map = {0: ['Co', 'Ni'], 4: ['Fe']}
            sub_map = {}
            rules = [r.strip() for r in sub_rule_str.split(';') if r.strip()]
            
            if not rules:
                return {}, "Empty rules"

            for rule in rules:
                if ':' not in rule: continue
                idx_str, elems_str = rule.split(':')
                try:
                    idx = int(idx_str.strip())
                    if idx < 0 or idx >= len(struct):
                        return {}, f"Index {idx} out of range"
                    
                    elems = [e.strip() for e in elems_str.split(',') if e.strip()]
                    if not elems: continue
                    
                    # 验证元素符号合法性 (可选)
                    for e in elems:
                        Element(e) # 如果非法会抛错
                        
                    sub_map[idx] = elems
                except Exception as e:
                    return {}, f"Parse Error in '{rule}': {str(e)}"

            if not sub_map:
                return {}, "No valid rules parsed"

            # 2. 生成组合 (笛卡尔积)
            # indices = [0, 4]
            # elem_lists = [['Co', 'Ni'], ['Fe']]
            # product -> ([Co, Fe], [Ni, Fe])
            sorted_indices = sorted(sub_map.keys())
            elem_lists = [sub_map[i] for i in sorted_indices]
            
            combinations = list(itertools.product(*elem_lists))
            
            # 3. 生成新结构
            count = 0
            for combo in combinations:
                # 复制原结构
                new_s = struct.copy()
                
                # 构建命名后缀
                name_parts = []
                
                # 执行替换
                for i, elem_sym in enumerate(combo):
                    site_idx = sorted_indices[i]
                    new_s.replace(site_idx, elem_sym)
                    name_parts.append(f"{site_idx}{elem_sym}")
                
                suffix = "_".join(name_parts)
                new_id = f"{parent_id}_sub_{suffix}"
                
                results[new_id] = {
                    "struct": new_s.as_dict(),
                    "meta": {
                        "Parent": parent_id,
                        "Info": f"Sub: {', '.join(name_parts)}",
                        "Formula": new_s.composition.reduced_formula
                    },
                    "type": "sub" # 标记为替换类型
                }
                count += 1
                
            return results, f"Generated {count} substitution structures"

        except Exception as e:
            return {}, f"Sub Error: {str(e)}"

    @staticmethod
    def run_relax(struct_dict):
        if not HAS_DEEPMD: return None, "DeepMD Not Installed"
        try:
            s = Structure.from_dict(struct_dict)
            atoms = AseAtomsAdaptor.get_atoms(s)
            if len(atoms) > 4: 
                z_pos = atoms.positions[:, 2]
                c = FixAtoms(indices=[i for i, z in enumerate(z_pos) if z < np.min(z_pos)+2.0])
                atoms.set_constraint(c)
            atoms.calc = DP(model=MODEL_PATH)
            traj_e = []
            dyn = BFGS(atoms)
            dyn.attach(lambda: traj_e.append(atoms.get_potential_energy()), interval=1)
            dyn.run(fmax=0.05, steps=50)
            return {"energy": traj_e, "struct": AseAtomsAdaptor.get_structure(atoms).as_dict()}, "Success"
        except Exception as e: return None, str(e)

    @staticmethod
    def submit_aims_calculation(struct_dict, structure_name, config_dict, aims_params=None, slurm_params=None, control_template=None, slurm_template=None):
        """
        提交 FHI-aims 计算
        """
        try:
            from services.aims.workflow import AimsWorkflowManager

            workflow = AimsWorkflowManager(config_dict)
            ok, msg = workflow.connect_remote()
            if not ok:
                return False, f"连接失败: {msg}"

            ok, files = workflow.prepare_calculation(
                struct_dict,
                structure_name,
                aims_params=aims_params,
                slurm_params=slurm_params,
                control_content=control_template,
                slurm_content=slurm_template
            )
            if not ok:
                workflow.disconnect_remote()
                return False, f"准备文件失败: {files.get('error', 'Unknown error')}"

            ok, result = workflow.submit_calculation(files)
            if ok:
                return True, {"job_id": result, "remote_dir": workflow.current_remote_dir, "status": "submitted"}
            workflow.disconnect_remote()
            return False, result
        except Exception as e:
            return False, f"AIMS提交错误: {str(e)}"

    @staticmethod
    def poll_aims_status(job_id, config_dict, check_interval=5, max_wait_seconds=1800):
        try:
            from services.aims.workflow import AimsWorkflowManager
            workflow = AimsWorkflowManager(config_dict)
            ok, msg = workflow.connect_remote()
            if not ok:
                return False, f"连接失败: {msg}"
            workflow.current_job_id = job_id
            ok, res = workflow.poll_and_get_results(job_id=job_id, check_interval=check_interval, max_wait_seconds=max_wait_seconds)
            workflow.disconnect_remote()
            if ok:
                return True, res
            return False, res.get("error", "Unknown error")
        except Exception as e:
            return False, f"轮询错误: {str(e)}"

    @staticmethod
    def fetch_completed_aims_results(job_id, remote_dir, config_dict):
        try:
            from services.aims.workflow import AimsWorkflowManager
            workflow = AimsWorkflowManager(config_dict)
            ok, msg = workflow.connect_remote()
            if not ok:
                return None
            ok, res = workflow.get_calculation_results(remote_dir, job_id)
            workflow.disconnect_remote()
            if ok:
                return res
            return None
        except Exception as e:
            print(f"AIMS fetch error: {e}")
            return None

    @staticmethod
    def download_final_aims_structure(remote_dir, config_dict, local_save_dir=None):
        try:
            from services.aims.workflow import AimsWorkflowManager
            import os
            from pathlib import Path
            workflow = AimsWorkflowManager(config_dict)
            ok, msg = workflow.connect_remote()
            if not ok:
                return False, f"连接失败: {msg}"
            if local_save_dir is None:
                root = config_dict.get("local_paths", {}).get("results_root") or "./outputs"
                local_save_dir = os.path.join(root, "high_throughput")
            Path(local_save_dir).mkdir(parents=True, exist_ok=True)
            ok, struct = workflow.fetch_final_structure(remote_dir)
            workflow.disconnect_remote()
            if ok and struct:
                return True, struct
            return False, "未获取到结构"
        except Exception as e:
            return False, f"下载错误: {str(e)}"

    @staticmethod
    def submit_vasp_calculation(struct_dict, structure_name, config_dict, vasp_params=None, slurm_params=None, incar_template=None, slurm_template=None):
        """
        提交VASP DFT计算到远程服务器
        
        Args:
            struct_dict: pymatgen Structure.as_dict() 结果
            structure_name: 结构标识符
            config_dict: 全局配置字典（包含remote_server等信息）
            vasp_params: VASP计算参数（可选，覆盖默认值）
            slurm_params: SLURM脚本参数（可选）
            incar_template: INCAR模板内容（可选）
            slurm_template: SLURM模板内容（可选）
            
        Returns:
            (成功标志, {'job_id': str, 'remote_dir': str} 或 错误信息)
        """
        try:
            from services.vasp.workflow import VaspWorkflowManager
            
            # 初始化工作流管理器
            workflow = VaspWorkflowManager(config_dict)
            
            # 连接到远程服务器
            success, msg = workflow.connect_remote()
            if not success:
                return False, f"连接失败: {msg}"
            
            # 准备计算文件
            success, calc_files = workflow.prepare_calculation(
                struct_dict,
                structure_name,
                vasp_params=vasp_params,
                slurm_params=slurm_params,
                incar_content=incar_template,
                slurm_content=slurm_template
            )
            if not success:
                workflow.disconnect_remote()
                return False, f"准备文件失败: {calc_files.get('error', 'Unknown error')}"
            
            # 提交计算任务
            success, result = workflow.submit_calculation(calc_files, use_default_potcar=False)
            
            if success:
                return True, {
                    'job_id': result,
                    'remote_dir': workflow.current_remote_dir,
                    'status': 'submitted'
                }
            else:
                workflow.disconnect_remote()
                return False, f"提交失败: {result}"
            
        except Exception as e:
            return False, f"VASP提交错误: {str(e)}"

    @staticmethod
    def poll_vasp_status(job_id, config_dict, check_interval=5, max_wait_seconds=1800):
        """
        轮询VASP计算状态并获取结果
        
        Args:
            job_id: SLURM任务ID
            config_dict: 全局配置字典
            check_interval: 轮询间隔（秒）
            max_wait_seconds: 最大等待时间（秒）
            
        Returns:
            (成功标志, 结果字典 或 错误信息)
        """
        try:
            from services.vasp.workflow import VaspWorkflowManager
            
            workflow = VaspWorkflowManager(config_dict)
            success, msg = workflow.connect_remote()
            if not success:
                return False, f"连接失败: {msg}"
            
            workflow.current_job_id = job_id
            
            # 等待并获取结果
            success, results = workflow.poll_and_get_results(
                job_id=job_id,
                check_interval=check_interval,
                max_wait_seconds=max_wait_seconds
            )
            
            workflow.disconnect_remote()
            
            if success:
                return True, results
            else:
                return False, results.get('error', 'Unknown error')
                
        except Exception as e:
            return False, f"轮询错误: {str(e)}"

    @staticmethod
    def check_batch_status(job_ids, config_dict):
        """批量检查计算状态 (VASP/AIMS 通用)"""
        try:
            from services.remote_server import ssh_manager
            ssh_cfg = config_dict.get("remote_server", {})
            ssh = ssh_manager.SSHManager(**ssh_cfg)
            ok, msg = ssh.connect()
            if not ok:
                return {}
            ssh.open_sftp()
            results = {}
            for jid in job_ids:
                exists, status = ssh.query_slurm_status(jid)
                results[jid] = "RUNNING" if exists else "COMPLETED"
            ssh.close()
            return results
        except Exception as e:
            print(f"Batch Check Error: {e}")
            return {}

    @staticmethod
    def fetch_completed_results(job_id, remote_dir, config_dict):
        """
        获取已完成任务的详细结果
        """
        try:
            from services.vasp.workflow import VaspWorkflowManager
            
            workflow = VaspWorkflowManager(config_dict)
            success, msg = workflow.connect_remote()
            if not success:
                return None
            
            success, results = workflow.get_calculation_results(remote_dir, job_id)
            workflow.disconnect_remote()
            
            if success:
                return results
            return None
        except Exception as e:
            print(f"Fetch Results Error: {e}")
            return None
    
    @staticmethod
    def download_final_structure(remote_dir, config_dict, local_save_dir=None):
        """
        从远程服务器下载最终结构(CONTCAR)并解析为Structure对象
        
        Args:
            remote_dir: 远程工作目录
            config_dict: 全局配置字典
            local_save_dir: 本地保存目录
            
        Returns:
            (成功标志, Structure对象 或 错误信息)
        """
        try:
            from services.vasp.workflow import VaspWorkflowManager
            import os
            from pathlib import Path
            
            workflow = VaspWorkflowManager(config_dict)
            success, msg = workflow.connect_remote()
            if not success:
                return False, f"连接失败: {msg}"
            
            workflow.current_remote_dir = remote_dir
            
            # 统一输出目录
            if local_save_dir is None:
                root = config_dict.get("local_paths", {}).get("results_root") or "./outputs"
                local_save_dir = os.path.join(root, "high_throughput")

            Path(local_save_dir).mkdir(parents=True, exist_ok=True)
            local_contcar = os.path.join(local_save_dir, "CONTCAR")
            
            # 下载CONTCAR
            success, msg = workflow.ssh.download_result_file(
                f"{remote_dir}/CONTCAR",
                local_contcar
            )
            
            workflow.disconnect_remote()
            
            if not success:
                return False, f"下载失败: {msg}"
            
            # 解析为Structure
            try:
                struct = Structure.from_file(local_contcar)
                return True, struct
            except Exception as e:
                return False, f"解析CONTCAR失败: {str(e)}"
        
        except Exception as e:
            return False, f"下载错误: {str(e)}"

# ================= 3. UI Setup =================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css"])
server = app.server
if HAS_CTC: ctc.register_app(app)

app.layout = dbc.Container([
    dcc.Store(id='store-data', data={}),
    dcc.Store(id='pt-selection', data=[]), 
    dcc.Download(id="download-manager"),
    dcc.Interval(id='interval-dft-monitor', interval=10000, n_intervals=0),

    dbc.NavbarSimple(
        brand="🧪 Integrated AI Catalyst Workbench",
        color="white", className="mb-3 shadow-sm",
        children=[dbc.NavItem(dbc.NavLink("Reset", href="/", external_link=True))]
    ),

    dbc.Row([
        # --- Left ---
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("1. Data Source", className="fw-bold text-primary"),
                dbc.CardBody([
                    dbc.Label("Option A: Search MP", className="fw-bold small"),
                    dbc.Input(id="api-key", placeholder="MP Key", value=DEFAULT_MP_KEY, type="password", size="sm", className="mb-1"),
                    dbc.InputGroup([
                        dbc.Input(id="search-input", placeholder="Li-Fe, CeO2...", size="sm"),
                        dbc.Button("⚛️", id="btn-open-pt", color="secondary", outline=True, size="sm"),
                    ], className="mb-2"),
                    dbc.Button("Search", id="btn-search", color="primary", size="sm", className="w-100 mb-3"),
                    html.Hr(),
                    dbc.Label("Option B: Upload (CIF/POSCAR)", className="fw-bold small"),
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            html.I(className="bi bi-cloud-upload me-2"),
                            "Drag & Drop Files"
                        ]),
                        className="upload-container",
                        style={"padding": "15px", "fontSize": "14px"},
                        multiple=True
                    ),
                    html.Div(id="msg-search", className="text-muted small mt-2 text-truncate")
                ])
            ], className="mb-3"),
            
            dbc.Card([
                dbc.CardHeader("2. Operations", className="fw-bold text-success"),
                dbc.CardBody([
                    # --- Slice ---
                    dbc.Label("1. Slice (h k l)", className="small fw-bold"),
                    dbc.Row([
                        dbc.Col(dbc.Input(id="h", placeholder="h", value=1, type="number", size="sm")),
                        dbc.Col(dbc.Input(id="k", placeholder="k", value=1, type="number", size="sm")),
                        dbc.Col(dbc.Input(id="l", placeholder="l", value=1, type="number", size="sm")),
                    ], className="mb-2"),
                    dbc.Label("Size Scale", className="small"),
                    dcc.Slider(id="sz", min=5, max=12, value=8, step=1, marks={5:'5', 12:'12'}),
                    dbc.Button("Generate Slab", id="btn-slice", color="success", outline=True, size="sm", className="w-100 mt-2 mb-3"),
                    
                    html.Hr(),
                    
                    # --- Adsorbate ---
                    dbc.Label("2. Adsorbate Settings", className="small fw-bold"),
                    dbc.Row([
                        dbc.Col([dbc.Label("Molecule", className="small text-muted mb-0"), dcc.Dropdown(id="mol-sel", options=["CO", "H2O", "O2", "N2", "OH"], value="CO", className="mb-1")]),
                    ]),
                    dbc.Row([
                        dbc.Col([dbc.Label("Search Mode", className="small text-muted mb-0"), dcc.Dropdown(id="ads-search-mode", options=[{"label": "Enumerate Sites", "value": "enum"}, {"label": "Genetic Algorithm", "value": "ga"}, {"label": "Min-hop (random)", "value": "minhop"}], value="enum", className="mb-1")])
                    ]),
                    dbc.Row([
                        dbc.Col([dbc.Label("GA Pop", className="small text-muted mb-0"), dbc.Input(id="ga-pop", type="number", value=12, min=4, step=2, size="sm", className="mb-1")], width=4),
                        dbc.Col([dbc.Label("GA Gen", className="small text-muted mb-0"), dbc.Input(id="ga-gen", type="number", value=10, min=1, step=1, size="sm", className="mb-1")], width=4),
                        dbc.Col([dbc.Label("Top K", className="small text-muted mb-0"), dbc.Input(id="ads-topk", type="number", value=5, min=1, step=1, size="sm", className="mb-1")], width=4)
                    ]),
                    dbc.Row([
                        dbc.Col([dbc.Label("Site Type", className="small text-muted mb-0"), dcc.Dropdown(id="site-type-sel", options=[{'label': 'All Sites', 'value': 'all'}, {'label': 'Ontop', 'value': 'ontop'}, {'label': 'Bridge', 'value': 'bridge'}, {'label': 'Hollow', 'value': 'hollow'}], value="all", className="mb-1")])
                    ]),
                    dbc.Row([
                        dbc.Col([dbc.Label("Target Atom Index (Opt)", className="small text-muted mb-0"), dbc.Input(id="target-atom-idx", placeholder="e.g. 15", type="number", size="sm", className="mb-2")])
                    ]),
                    dbc.Button("Add Molecule", id="btn-ads", color="info", outline=True, size="sm", className="w-100 mb-3"),

                    html.Hr(),

                    # --- New: Substitution ---
                    dbc.Label("3. HT Substitution", className="small fw-bold text-warning"),
                    dbc.Label("Format: Idx:El1,El2; Idx2:El3...", className="small text-muted mb-1", style={"fontSize": "10px"}),
                    dbc.Input(id="sub-input", placeholder="e.g. 0:Co,Ni; 12:Fe", size="sm", className="mb-2"),
                    dbc.Button("Run Substitution", id="btn-sub", color="warning", outline=True, size="sm", className="w-100"),
                    
                    # --- New: Reaction Pathway ---
                    dbc.Label("4. Reaction Pathway", className="small fw-bold text-info mt-2"),
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(
                                id="rxn-type-dropdown",
                                options=[
                                    {"label": "N2RR (Nitrogenase)", "value": "N2RR"},
                                    {"label": "ORR (Oxygen Reduction)", "value": "ORR"},
                                    {"label": "CORR (CO2 Reduction)", "value": "CORR"},
                                ],
                                placeholder="Select pathway",
                                clearable=True,
                                style={"fontSize": "12px"}
                            )
                        ], width=12)
                    ]),
                    dbc.Input(id="rxn-atom-idx", placeholder="Target atom index", type="number", size="sm", className="mb-2 mt-2"),
                    dbc.Button("Add Intermediates", id="btn-rxn", color="info", outline=True, size="sm", className="w-100"),
                ])
            ], className="mb-3"),
        ], width=3),

        # --- Middle ---
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([dbc.Row([dbc.Col("List", width=8), dbc.Col(dbc.Button("ZIP", id="btn-dl-all", size="sm", color="secondary", className="float-end"), width=4)], align="center")]),
                dbc.CardBody([
                    dash_table.DataTable(
                        id='main-table',
                        columns=[
                            {"name": "ID", "id": "id"}, 
                            {"name": "Info", "id": "info"},
                            {"name": "Energy", "id": "energy"},
                            {"name": "Converged", "id": "converged"}
                        ],
                        data=[], row_selectable="multi", selected_rows=[],
                        style_table={'height': '75vh', 'overflowY': 'auto'},
                        style_cell={'textAlign': 'left', 'fontSize': '12px', 'padding': '5px'},
                        style_data_conditional=[
                            {'if': {'filter_query': '{type} = "bulk"'}, 'color': '#007bff'},
                            {'if': {'filter_query': '{type} = "slab"'}, 'color': '#28a745'},
                            {'if': {'filter_query': '{type} = "ads"'}, 'color': '#17a2b8'},
                            {'if': {'filter_query': '{type} = "sub"'}, 'color': '#ffc107'}, # 替换结构颜色
                            {'if': {'filter_query': '{type} = "dft_result"'}, 'color': '#6f42c1'}, # DFT结果颜色
                        ]
                    )
                ], className="p-0")
            ], style={"height": "100%"})
        ], width=4),

        # --- Right ---
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Preview"),
                dbc.CardBody([
                    dcc.Loading(html.Div(id="struct-viewer", style={"height": "500px"}), type="cube"), 
                    html.Div(id="viewer-meta", className="mt-2 text-center text-primary")
                ])
            ], className="mb-3"),
            dbc.Card([
                dbc.CardHeader("DeepMD Relax"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(dbc.Button("Run", id="btn-calc", color="danger", className="w-100", disabled=not HAS_DEEPMD), width=3),
                        dbc.Col(dcc.Graph(id="graph-calc", style={"height": "100px"}, config={'displayModeBar': False}), width=9)
                    ])
                ])
            ], className="mb-3"),
            
            # === DFT/VASP Calculation Panel ===
            dbc.Card([
                dbc.CardHeader("🔬 DFT (VASP) Calculation", className="fw-bold text-success"),
                dbc.CardBody([
                    dbc.Alert("Note: Requires remote server access", color="info", className="py-2"),

                    dbc.Label("Engine", className="small fw-bold"),
                    dcc.Dropdown(
                        id="dft-engine",
                        options=[{"label": "VASP", "value": "vasp"}, {"label": "FHI-aims", "value": "aims"}],
                        value="vasp",
                        clearable=False,
                        className="mb-2"
                    ),
                    
                    # === Template Selection ===
                    dbc.Label("Calculation & Queue Templates", className="small fw-bold"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("VASP Calculation Type", className="text-muted small"),
                            dcc.Dropdown(
                                id="dft-calc-template",
                                options=[{"label": VaspCalculationTemplates.get_template_description(t), "value": t} 
                                        for t in VaspCalculationTemplates.get_template_names()],
                                value="geometry_opt",
                                clearable=False
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Queue System", className="text-muted small"),
                            dcc.Dropdown(
                                id="dft-queue-template",
                                options=[{"label": QueueSystemTemplates.get_template_description(q), "value": q}
                                        for q in QueueSystemTemplates.get_queue_systems()],
                                value="slurm",
                                clearable=False
                            )
                        ], width=6)
                    ]),
                    
                    dbc.Row([
                        dbc.Col(dbc.Button("📝 Edit VASP Template", id="btn-edit-vasp-template", color="warning", size="sm", className="w-100"), width=6),
                        dbc.Col(dbc.Button("📝 Edit Queue Template", id="btn-edit-queue-template", color="warning", size="sm", className="w-100"), width=6)
                    ], className="mt-2"),
                    
                    html.Hr(className="my-2"),
                    
                    dbc.Row([
                        dbc.Col(dbc.Button("📤 Submit to VASP", id="btn-dft-submit", color="success", className="w-100", disabled=not HAS_CONFIG), width=8),
                        dbc.Col(dbc.Button("🔄 拉取状态", id="btn-dft-poll", color="info", className="w-100", disabled=True), width=4)
                    ]),
                    html.Div(id="dft-msg", className="text-muted small mt-2")
                ])
            ]),
            
            # === Reaction Diagram Panel ===
            dbc.Card([
                dbc.CardHeader("📈 Reaction Energy Diagram", className="fw-bold text-primary"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Reaction Type", className="small"),
                            dcc.Dropdown(
                                id="step-graph-type",
                                options=[
                                    {"label": "NO2RR (Path 1-4)", "value": "NO2RR"},
                                    {"label": "NO2RR1 (Path 1-4)", "value": "NO2RR1"},
                                    {"label": "HER", "value": "HER"},
                                    {"label": "OER", "value": "OER"},
                                    {"label": "CO2RR", "value": "CO2RR"},
                                    {"label": "ORR", "value": "ORR"}
                                ],
                                value="NO2RR",
                                clearable=False
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Action", className="small"),
                            dbc.Button("Draw Diagram", id="btn-draw-step", color="primary", size="sm", className="w-100")
                        ], width=6)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Parameters", className="small"),
                            dbc.InputGroup([
                                dbc.InputGroupText("U (V)"),
                                dbc.Input(id="step-graph-U", type="number", value=0.0, step=0.1),
                                dbc.InputGroupText("pH"),
                                dbc.Input(id="step-graph-pH", type="number", value=0.0, step=1.0),
                            ], size="sm")
                        ], width=12)
                    ], className="mt-2"),
                    html.Div(id="step-graph-container", className="mt-3 text-center")
                ])
            ], className="mt-3")
        ], width=5)
    ], className="g-3 my-2"),

    # === Periodic Table Modal ===
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Select Elements")),
        dbc.ModalBody([
            create_periodic_table(), 
            html.Div(id="pt-selected-display", className="mt-2 fw-bold text-center text-primary"),
        ]),
        dbc.ModalFooter([
            dbc.Button("Clear", id="pt-clear", color="secondary", outline=True, size="sm"),
            dbc.Button("Apply", id="pt-apply", color="primary", size="sm")
        ])
    ], id="pt-modal", size="xl", is_open=False),

    # === Template Editor Modal (VASP) ===
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Edit VASP Calculation Template")),
        dbc.ModalBody([
            dbc.Alert("Changes are only effective for this session", color="warning", className="py-2"),
            html.Div(id="vasp-template-editor-container", children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Calculation Type", className="fw-bold"),
                        dcc.Dropdown(id="vasp-template-select", value="geometry_opt", clearable=False)
                    ])
                ]),
                html.Br(),
                dbc.Label("Template Content", className="fw-bold"),
                dcc.Textarea(
                    id="vasp-template-textarea",
                    placeholder="VASP template content will appear here...",
                    style={"width": "100%", "height": "400px", "fontFamily": "monospace", "fontSize": "11px"},
                    className="form-control"
                )
            ])
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="btn-vasp-template-cancel", color="secondary", outline=True),
            dbc.Button("Reset to Default", id="btn-vasp-template-reset", color="warning", outline=True),
            dbc.Button("Save Changes", id="btn-vasp-template-save", color="primary")
        ])
    ], id="vasp-template-modal", size="xl", is_open=False),

    # === Template Editor Modal (Queue) ===
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Edit Queue System Template")),
        dbc.ModalBody([
            dbc.Alert("Changes are only effective for this session", color="warning", className="py-2"),
            html.Div(id="queue-template-editor-container", children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Queue System", className="fw-bold"),
                        dcc.Dropdown(id="queue-template-select", value="slurm", clearable=False)
                    ])
                ]),
                html.Br(),
                dbc.Label("Template Content", className="fw-bold"),
                dcc.Textarea(
                    id="queue-template-textarea",
                    placeholder="Queue template content will appear here...",
                    style={"width": "100%", "height": "400px", "fontFamily": "monospace", "fontSize": "11px"},
                    className="form-control"
                )
            ])
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="btn-queue-template-cancel", color="secondary", outline=True),
            dbc.Button("Reset to Default", id="btn-queue-template-reset", color="warning", outline=True),
            dbc.Button("Save Changes", id="btn-queue-template-save", color="primary")
        ])
    ], id="queue-template-modal", size="xl", is_open=False),

    # === Template Operation Results ===
    dcc.Store(id="template-editor-message"),

], fluid=True, style={"height": "100vh"})

# ================= 4. Callbacks =================
@app.callback(
    Output("pt-modal", "is_open"),
    Output("search-input", "value"),
    Output("pt-selection", "data"),
    Output("pt-selected-display", "children"),
    Input("btn-open-pt", "n_clicks"),
    Input("pt-apply", "n_clicks"),
    Input("pt-clear", "n_clicks"),
    Input({"type": "pt-btn", "elem": ALL}, "n_clicks"),
    State("pt-modal", "is_open"),
    State("pt-selection", "data"),
    State("search-input", "value"),
    prevent_initial_call=True
)
def handle_pt(n_open, n_apply, n_clear, _, is_open, current_sel, current_text):
    trigger = ctx.triggered_id
    if trigger == "btn-open-pt": return True, no_update, no_update, no_update
    if isinstance(trigger, dict) and trigger.get("type") == "pt-btn":
        elem = trigger["elem"]
        new_sel = current_sel.copy()
        if elem in new_sel: new_sel.remove(elem)
        else: new_sel.append(elem)
        return no_update, no_update, new_sel, "Selected: " + ", ".join(new_sel)
    if trigger == "pt-clear": return no_update, no_update, [], "Selected: None"
    if trigger == "pt-apply": return False, ", ".join(current_sel), current_sel, no_update
    return no_update, no_update, no_update, no_update

@app.callback(
    Output("step-graph-container", "children"),
    Input("btn-draw-step", "n_clicks"),
    State("main-table", "selected_rows"),
    State("main-table", "data"),
    State("store-data", "data"),
    State("step-graph-type", "value"),
    State("step-graph-U", "value"),
    State("step-graph-pH", "value"),
    prevent_initial_call=True
)
def draw_reaction_step(n_clicks, sel_rows, tbl_data, store, rxn_type, U, pH):
    if not n_clicks or not sel_rows or not store:
        return dbc.Alert("Please select a parent structure first", color="warning")
    
    try:
        # 1. Identify Unique Parent Structures from Selection
        # We iterate through ALL selected rows to support multi-path comparison.
        unique_parents = set()
        
        for idx in sel_rows:
            sel_id = tbl_data[idx]["id"]
            sel_item = store.get(sel_id)
            if not sel_item: continue
            
            # Find root parent (Slab ID)
            current_meta = sel_item.get("meta", {})
            parent_id = sel_id # Default assumption
            
            if "Parent" in current_meta:
                curr = sel_item
                while "Parent" in curr.get("meta", {}) and curr["meta"]["Parent"] in store:
                    curr_id = curr["meta"]["Parent"]
                    curr = store[curr_id]
                    parent_id = curr_id
            
            unique_parents.add(parent_id)
        
        if not unique_parents:
            return dbc.Alert("No valid parent structures found in selection", color="warning")

        # 2. Collect Data Points for EACH Parent
        temp_files = []
        import tempfile
        import os
        
        try:
            for parent_id in unique_parents:
                data_points = {} # Name -> Energy
                
                # Helper to get origin of a result
                def get_origin(res_item):
                    if "Parent" in res_item.get("meta", {}):
                        return res_item["meta"]["Parent"]
                    return None

                # Helper to check if an item is an intermediate of parent_id for rxn_type
                def is_intermediate(item_id):
                    item = store.get(item_id)
                    if not item: return False, None
                    meta = item.get("meta", {})
                    if meta.get("Parent") == parent_id:
                        if meta.get("Reaction") == rxn_type:
                            return True, meta.get("Molecule")
                    return False, None

                # Scan store for this parent
                for uid, item in store.items():
                    if item.get("type") != "dft_result": continue
                    
                    origin_id = get_origin(item)
                    if not origin_id: continue
                    
                    # Check if this result is for the clean slab
                    if origin_id == parent_id:
                        try:
                            e_str = item["meta"].get("Energy", "").replace(" eV", "")
                            if e_str: data_points["*"] = float(e_str)
                        except: pass
                        continue
                        
                    # Check if this result is for an intermediate
                    is_inter, mol_name = is_intermediate(origin_id)
                    if is_inter and mol_name:
                        try:
                            e_str = item["meta"].get("Energy", "").replace(" eV", "")
                            if e_str: data_points[mol_name] = float(e_str)
                        except: pass

                if "*" not in data_points:
                    # Skip parents without clean slab energy, or maybe warn?
                    # For multi-plot, we just skip to avoid breaking everything.
                    print(f"Skipping {parent_id}: Missing clean slab energy (*)")
                    continue
                    
                if len(data_points) < 2:
                     print(f"Skipping {parent_id}: Not enough data points")
                     continue

                # 3. Generate Input File for draw_step.py
                # The first line's first word is used as the legend label.
                # We use parent_id (shortened) as the label.
                short_label = parent_id[:15] 
                
                lines = [f"{short_label}\tE0\tZPE\tH\tTS\tGcor\tG"]
                for name, energy in data_points.items():
                    lines.append(f"{name}\t{energy}\t0\t0\t0\t0\t{energy}")
                    
                input_content = "\n".join(lines)
                
                fd, temp_path = tempfile.mkstemp(suffix=".txt", text=True)
                with os.fdopen(fd, 'w') as tmp:
                    tmp.write(input_content)
                temp_files.append(temp_path)
            
            if not temp_files:
                return dbc.Alert("No valid data found for any selected structure (Check if calculations are complete and include '*')", color="warning")

            # 4. Call step_graph with LIST of files
            rxn_dict_map = {
                "NO2RR": NO2RR, "NO2RR1": NO2RR1, "HER": HER, 
                "OER": OER, "CO2RR": CO2RR, "ORR": ORR
            }
            rxn_obj = rxn_dict_map.get(rxn_type, NO2RR)
            
            # If multiple files, step_graph uses the list logic.
            # If single file, we pass it as a string (or list of length 1? draw_step handles list even if len=1 in else block? No, it checks isinstance(str))
            # Let's pass list if len > 1, else string.
            
            target_arg = temp_files if len(temp_files) > 1 else temp_files[0]
            
            # Note: step_graph saves file to f'{filenames}_U_{U}_pH_{pH}.png'
            # If filenames is a list, it uses str(filenames) which is ugly: "['...tmp1', '...tmp2']_U_...png"
            # We need to be careful about the output filename.
            
            # Workaround: We will patch plt.savefig temporarily to capture the figure
            # or we can just let it save to the weird filename and try to read it.
            # The weird filename will be something like "['C:\\Temp\\tmp1', 'C:\\Temp\\tmp2']_U_0.0_pH_0.0.png"
            # This is a valid filename on Windows? No, brackets and quotes are not valid in filenames usually?
            # Actually, on Windows, quotes " are illegal. Brackets [] are legal.
            # But str(['path']) contains quotes. So it will fail on Windows.
            
            # Solution: We MUST modify draw_step.py to handle list filenames gracefully for saving.
            # OR we can pass a single string if len=1.
            # If len > 1, we are stuck unless we edit draw_step.py.
            
            # Let's assume the user is okay with me editing draw_step.py to fix this bug.
            # But first, let's try to run it. If it fails, we know why.
            
            # Actually, I can just patch plt.savefig in THIS process before calling step_graph.
            # This is safer and cleaner than editing the utility file.
            
            # Create a known output path
            out_img_path = temp_files[0] + "_out.png"
            
            original_savefig = plt.savefig
            def patched_savefig(*args, **kwargs):
                # Ignore the filename passed by step_graph and use our own
                original_savefig(out_img_path, bbox_inches='tight')
                
            plt.savefig = patched_savefig
            
            try:
                step_graph(target_arg, rxn_obj, '*', U=float(U), pH=float(pH))
            finally:
                # Restore original savefig
                plt.savefig = original_savefig
            
            if os.path.exists(out_img_path):
                # Read image and encode to base64
                with open(out_img_path, "rb") as img_f:
                    encoded_image = base64.b64encode(img_f.read()).decode('ascii')
                
                # Cleanup output image
                try: os.remove(out_img_path)
                except: pass
                
                return html.Img(src=f"data:image/png;base64,{encoded_image}", style={"maxWidth": "100%"})
            else:
                return dbc.Alert("Failed to generate plot image", color="danger")
                
        finally:
            # Cleanup temp input files
            for f in temp_files:
                if os.path.exists(f):
                    try: os.remove(f)
                    except: pass
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return dbc.Alert(f"Error generating diagram: {str(e)}", color="danger")


@app.callback(
    Output("store-data", "data"),
    Output("main-table", "data"),
    Output("main-table", "selected_rows"), 
    Output("msg-search", "children"),
    Output("dft-msg", "children"),
    Output("btn-dft-poll", "disabled"),
    Input("btn-search", "n_clicks"),
    Input("upload-data", "contents"),
    Input("btn-slice", "n_clicks"),
    Input("btn-ads", "n_clicks"),
    Input("btn-sub", "n_clicks"),
    Input("btn-rxn", "n_clicks"),
    Input("btn-dft-submit", "n_clicks"),
    Input("btn-dft-poll", "n_clicks"),
    Input("interval-dft-monitor", "n_intervals"),
    State("upload-data", "filename"),
    State("store-data", "data"),
    State("main-table", "selected_rows"),
    State("main-table", "data"),
    State("api-key", "value"), State("search-input", "value"),
    State("h", "value"), State("k", "value"), State("l", "value"), State("sz", "value"),
    State("mol-sel", "value"), State("site-type-sel", "value"), State("target-atom-idx", "value"),
    State("ads-search-mode", "value"), State("ga-pop", "value"), State("ga-gen", "value"), State("ads-topk", "value"),
    State("sub-input", "value"),
    State("rxn-type-dropdown", "value"), State("rxn-atom-idx", "value"),
    State("dft-calc-template", "value"),
    State("dft-queue-template", "value"),
    State("dft-engine", "value"),
    prevent_initial_call=True
)
def main_flow(b_s, up_contents, b_sl, b_ad, b_sub, b_rxn, b_dft_sub, b_dft_poll, n_dft_interval, up_filenames, store, sel, table, key, query, h, k, l, sz, mol, site_type, atom_idx, ads_mode, ga_pop, ga_gen, ads_topk, sub_rule, rxn_type, rxn_atom_idx, vasp_template_name, queue_template_name, dft_engine):
    trigger = ctx.triggered_id
    new_store = store.copy() if store else {}
    msg = ""
    dft_msg = no_update
    poll_disabled = no_update

    # 辅助函数：更新表格数据
    def update_table_data(current_store):
        def sort_key(uid): 
            t = current_store[uid].get("type", "unknown")
            return (0 if t=="bulk" else 1 if t=="slab" else 2 if t=="ads" else 3 if t=="ads_rxn" else 4, uid)
        
        new_tbl = []
        for u, v in sorted(current_store.items(), key=lambda x: sort_key(x[0])):
            meta = v.get("meta", {})
            new_tbl.append({
                "id": u,
                "type": v.get("type", "unknown"),
                "info": meta.get("Info", meta.get("Formula", "")),
                "energy": meta.get("Energy", ""),
                "converged": meta.get("Converged", "")
            })
        return new_tbl

    if trigger == "btn-search":
        if not query: return no_update, no_update, no_update, "Please enter query", no_update, no_update
        data, err = OCPLogic.smart_search_mp(key, query)
        if not data: return no_update, no_update, no_update, err, no_update, no_update
        new_store = data
        msg = f"Found {len(data)} items"
    
    elif trigger == "upload-data" and up_contents:
        count = 0
        for content, fname in zip(up_contents, up_filenames):
            res, err = OCPLogic.parse_upload(content, fname)
            if res:
                new_store.update(res)
                count += 1
            else: msg = f"Error: {err}"
        if count > 0: msg = f"Uploaded {count} files"

    elif trigger == "btn-slice" and sel:
        ids = [table[i]["id"] for i in sel]
        count = 0
        for tid in ids:
            item = store.get(tid)
            if item and item["type"] == "bulk":
                new_store.update(OCPLogic.generate_slabs(tid, item["struct"], h, k, l, sz))
                count += 1
        msg = f"Generated {count} slabs"

    elif trigger == "btn-ads" and sel:
        ids = [table[i]["id"] for i in sel]
        count = 0
        target_idx = None
        if atom_idx is not None and str(atom_idx).strip() != "":
            target_idx = int(atom_idx)

        for tid in ids:
            item = store.get(tid)
            if item and item["type"] == "slab":
                res = OCPLogic.search_adsorption(
                    tid,
                    item["struct"],
                    mol,
                    mode=ads_mode,
                    target_atom_idx=target_idx,
                    ga_params={"pop": ga_pop or 12, "gen": ga_gen or 10},
                    top_k=ads_topk or 5
                )
                if res:
                    new_store.update(res)
                    count += len(res)
        msg = f"Added {count} adsorption structures via {ads_mode or 'enum'}"

    elif trigger == "btn-sub" and sel:
        if not sub_rule:
            return no_update, no_update, no_update, "Please enter substitution rules", no_update, no_update
        
        ids = [table[i]["id"] for i in sel]
        total_count = 0
        last_msg = ""
        
        for tid in ids:
            item = store.get(tid)
            if item:
                res, info = OCPLogic.generate_substitutions(tid, item["struct"], sub_rule)
                if res:
                    new_store.update(res)
                    total_count += len(res)
                else:
                    last_msg = info
        
        if total_count > 0:
            msg = f"Generated {total_count} substituted structures"
        else:
            msg = f"Substitution failed: {last_msg}"

    elif trigger == "btn-rxn" and sel:
        if not rxn_type:
            return no_update, no_update, no_update, "Please select reaction pathway", no_update, no_update
        if rxn_atom_idx is None:
            return no_update, no_update, no_update, "Please enter target atom index", no_update, no_update
        
        ids = [table[i]["id"] for i in sel]
        total_count = 0
        last_msg = ""
        
        for tid in ids:
            item = store.get(tid)
            if item:
                res, info = OCPLogic.generate_reaction_intermediates(tid, item["struct"], rxn_type, int(rxn_atom_idx))
                if res:
                    new_store.update(res)
                    total_count += len(res)
                else:
                    last_msg = info
        
        if total_count > 0:
            msg = f"Generated {total_count} {rxn_type} intermediate structures"
        else:
            msg = f"Pathway failed: {last_msg}"

    # === DFT Submit Logic ===
    elif trigger == "btn-dft-submit":
        if not HAS_CONFIG:
            dft_msg = "❌ 配置未加载，无法连接远程服务器"
            poll_disabled = True
        elif not sel or not store or not table:
            dft_msg = "⚠️ 请先选择要计算的结构"
            poll_disabled = True
        else:
            try:
                engine = dft_engine or "vasp"
                selected_ids = [table[i]["id"] for i in sel]
                successful_jobs = []
                failed_jobs = []
                
                for struct_id in selected_ids:
                    if struct_id not in store:
                        failed_jobs.append(f"{struct_id} (未找到)")
                        continue
                    structure = store[struct_id]

                    if engine == "vasp":
                        vasp_params = {
                            "encut": 500, "nsw": 100, "ibrion": 2, "ediff": 1e-5, "ismear": 0,
                            "sigma": 0.05, "nelm": 100, "nelmin": 4, "potim": 0.5, "prec": "High"
                        }
                        base_slurm_params = {
                            "n_nodes": 1, "n_procs": 16, "time_limit": "01:00:00", "partition": "gpu"
                        }
                        incar_content = SESSION_EDITOR.get_vasp_template_with_params(vasp_template_name, vasp_params)
                        current_slurm_params = base_slurm_params.copy()
                        current_slurm_params["job_name"] = f"VASP_{struct_id[:20]}"
                        slurm_content = SESSION_EDITOR.get_queue_template_with_params(queue_template_name, current_slurm_params)
                        success, result = OCPLogic.submit_vasp_calculation(
                            structure["struct"],
                            struct_id,
                            CONFIG.config if hasattr(CONFIG, 'config') else {},
                            vasp_params=vasp_params,
                            slurm_params=current_slurm_params,
                            incar_template=incar_content,
                            slurm_template=slurm_content
                        )
                    else:
                        aims_params = {"kgrid": [4, 4, 1]}
                        slurm_params = {"n_nodes": 1, "n_procs": 16, "time_limit": "02:00:00", "partition": "cpu"}
                        control_tmpl = AimsTemplates.get_template(vasp_template_name) or AimsTemplates.get_template("geometry_opt")
                        success, result = OCPLogic.submit_aims_calculation(
                            structure["struct"],
                            struct_id,
                            CONFIG.config if hasattr(CONFIG, 'config') else {},
                            aims_params=aims_params,
                            slurm_params=slurm_params,
                            control_template=control_tmpl,
                            slurm_template=None
                        )

                    if success:
                        job_id = result['job_id']
                        dft_job_tracker[job_id] = {
                            'structure_id': struct_id,
                            'remote_dir': result['remote_dir'],
                            'submitted_time': datetime.datetime.now().isoformat(),
                            'status': 'submitted',
                            'template_name': vasp_template_name,
                            'engine': engine
                        }
                        successful_jobs.append((struct_id, job_id))
                    else:
                        failed_jobs.append(f"{struct_id} ({result})")
                
                msg_lines = []
                if successful_jobs:
                    msg_lines.append(f"✅ 成功提交 {len(successful_jobs)} 个任务:")
                    for struct_id, job_id in successful_jobs:
                        msg_lines.append(f"  • {struct_id}: {job_id}")
                if failed_jobs:
                    msg_lines.append(f"\n❌ 提交失败 {len(failed_jobs)} 个:")
                    for fail_info in failed_jobs:
                        msg_lines.append(f"  • {fail_info}")
                
                dft_msg = "\n".join(msg_lines) if msg_lines else "⚠️ 没有提交任何任务"
                poll_disabled = len(successful_jobs) == 0
            except Exception as e:
                dft_msg = f"❌ 错误: {str(e)}"
                poll_disabled = True

    # === DFT Poll Logic ===
    elif trigger in ["btn-dft-poll", "interval-dft-monitor"]:
        if not dft_job_tracker:
             dft_msg = "⚠️ 没有正在进行的任务"
             poll_disabled = True
        else:
            job_ids = list(dft_job_tracker.keys())
            status_map = OCPLogic.check_batch_status(job_ids, CONFIG.config if hasattr(CONFIG, 'config') else {})
            
            msg_lines = ["🔄 任务状态更新:"]
            active_jobs = 0
            updated_store = False
            
            for jid, status in status_map.items():
                dft_job_tracker[jid]['status'] = status
                struct_id = dft_job_tracker[jid]['structure_id']
                engine = dft_job_tracker[jid].get('engine', 'vasp')
                
                details = ""
                if status == "COMPLETED":
                    remote_dir = dft_job_tracker[jid]['remote_dir']
                    
                    # 1. 获取计算结果 (Energy, Time, etc.)
                    if 'results' not in dft_job_tracker[jid]:
                        if engine == 'aims':
                            res = OCPLogic.fetch_completed_aims_results(jid, remote_dir, CONFIG.config if hasattr(CONFIG, 'config') else {})
                        else:
                            res = OCPLogic.fetch_completed_results(jid, remote_dir, CONFIG.config if hasattr(CONFIG, 'config') else {})
                        if res:
                            dft_job_tracker[jid]['results'] = res
                    
                    # 2. 获取结构文件 (CONTCAR)
                    if 'final_struct_dict' not in dft_job_tracker[jid]:
                        if engine == 'aims':
                            success, struct_or_msg = OCPLogic.download_final_aims_structure(remote_dir, CONFIG.config if hasattr(CONFIG, 'config') else {})
                        else:
                            success, struct_or_msg = OCPLogic.download_final_structure(remote_dir, CONFIG.config if hasattr(CONFIG, 'config') else {})
                        if success:
                            dft_job_tracker[jid]['final_struct_dict'] = struct_or_msg.as_dict()
                            dft_job_tracker[jid]['formula'] = struct_or_msg.composition.reduced_formula
                    
                    # 3. 更新数据存储 (如果尚未更新且数据齐全)
                    # 检查是否已经处理过这个任务的结果写入
                    result_key = f"processed_{jid}"
                    if not dft_job_tracker[jid].get(result_key) and 'final_struct_dict' in dft_job_tracker[jid]:
                        try:
                            res = dft_job_tracker[jid].get('results', {})
                            final_struct_dict = dft_job_tracker[jid]['final_struct_dict']
                            final_formula = dft_job_tracker[jid].get('formula', 'Unknown')
                            
                            template_name = dft_job_tracker[jid].get('template_name', 'unknown')
                            prefix = "opt" if "opt" in template_name or "relax" in template_name else "sp"
                            new_id = f"{prefix}_{struct_id}"
                            
                            energy_val = ""
                            converged_val = ""
                            if engine == 'aims':
                                if 'energy' in res:
                                    energy_val = f"{res['energy']:.4f} eV"
                                    converged_val = "Yes"
                            elif 'oszicar' in res and res['oszicar'].get('converged'):
                                energy_val = f"{res['oszicar'].get('final_energy', 0):.4f} eV"
                                converged_val = "Yes"
                            elif 'outcar' in res:
                                converged_val = "Yes" if res['outcar'].get('converged') else "No"
                            
                            new_store[new_id] = {
                                "struct": final_struct_dict,
                                "meta": {
                                    "Formula": final_formula,
                                    "Info": final_formula,
                                    "Energy": energy_val,
                                    "Converged": converged_val,
                                    "Parent": struct_id,
                                    "Source": f"{engine.upper()} Calculation"
                                },
                                "type": "dft_result"
                            }
                            updated_store = True
                            dft_job_tracker[jid][result_key] = True # 标记为已处理，避免重复写入
                        except Exception as e:
                            print(f"Error writing back result: {e}")

                    if 'results' in dft_job_tracker[jid]:
                        res = dft_job_tracker[jid]['results']
                        energy = "N/A"
                        time_str = "N/A"
                        converged = "Unknown"
                        if engine == 'aims':
                            if 'energy' in res:
                                energy = f"{res['energy']:.4f} eV"
                                converged = "✅ Done"
                        elif 'oszicar' in res and res['oszicar'].get('converged'):
                            energy = f"{res['oszicar'].get('final_energy', 'N/A'):.4f} eV"
                            converged = "✅ Converged"
                        elif 'outcar' in res:
                             if res['outcar'].get('converged'): converged = "✅ Converged"
                             else: converged = "❌ Not Converged"
                        if 'outcar' in res:
                            t = res['outcar'].get('elapsed_time')
                            if t: time_str = f"{t:.1f}s"
                            
                        details = f"\n    └─ E={energy}, Time={time_str}, {converged}"

                msg_lines.append(f"  • {struct_id} ({jid}): {status}{details}")
                if status == "RUNNING":
                    active_jobs += 1
            
            dft_msg = "\n".join(msg_lines)
            poll_disabled = (active_jobs == 0)

    tbl = update_table_data(new_store)
    
    return new_store, tbl, [], msg, dft_msg, poll_disabled

@app.callback(
    Output("struct-viewer", "children"),
    Output("viewer-meta", "children"),
    Input("main-table", "selected_rows"),
    State("main-table", "data"),
    State("store-data", "data"),
    prevent_initial_call=True
)
def show_struct(idx, tbl, store):
    if not idx or not store: return html.Div("Please select an item"), ""
    if not HAS_CTC: return html.Div("Crystal Toolkit not installed"), "No Preview"
    uid = tbl[idx[-1]]["id"]
    item = store[uid]
    s = Structure.from_dict(item["struct"])
    return ctc.StructureMoleculeComponent(s, id="view_comp").layout(), f"{uid} | {item['meta'].get('Info', '')}"

@app.callback(
    Output("graph-calc", "figure"),
    Input("btn-calc", "n_clicks"),
    State("main-table", "selected_rows"),
    State("main-table", "data"),
    State("store-data", "data"),
    prevent_initial_call=True
)
def run_calc(n, idx, tbl, store):
    if not idx: raise PreventUpdate
    uid = tbl[idx[-1]]["id"]
    res, msg = OCPLogic.run_relax(store[uid]["struct"])
    if not res: return {"layout": {"title": msg}}
    return {"data": [{"y": res["energy"], "type": "line"}], "layout": {"margin": {"t":20,"b":20,"l":30,"r":10}}}

@app.callback(
    Output("download-manager", "data"),
    Input("btn-dl-all", "n_clicks"),
    State("store-data", "data"),
    prevent_initial_call=True
)
def dl_all(n, store):
    if not store: raise PreventUpdate
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for u, v in store.items(): 
            s = Structure.from_dict(v["struct"])
            z.writestr(f"{u}.cif", s.to(fmt="cif"))
    b.seek(0)
    return dcc.send_bytes(b.read(), "data.zip")

# === DFT Calculation Callbacks ===

# submit_dft_calc callback removed as it is merged into main_flow

# ================= Template Editor Callbacks =================

# 初始化 VASP 模板编辑器下拉选项
if HAS_CONFIG and SESSION_EDITOR:
    @app.callback(
        Output("vasp-template-select", "options"),
        Input("vasp-template-modal", "is_open"),
        prevent_initial_call=True
    )
    def init_vasp_templates(_):
        return [{"label": VaspCalculationTemplates.get_template_description(t), "value": t}
                for t in VaspCalculationTemplates.get_template_names()]


# 初始化队列模板编辑器下拉选项
if HAS_CONFIG and SESSION_EDITOR:
    @app.callback(
        Output("queue-template-select", "options"),
        Input("queue-template-modal", "is_open"),
        prevent_initial_call=True
    )
    def init_queue_templates(_):
        return [{"label": QueueSystemTemplates.get_template_description(q), "value": q}
                for q in QueueSystemTemplates.get_queue_systems()]


# 打开、选择和重置 VASP 模板编辑对话框（合并回调以避免重复输出）
@app.callback(
    Output("vasp-template-modal", "is_open"),
    Output("vasp-template-textarea", "value"),
    Input("btn-edit-vasp-template", "n_clicks"),
    Input("btn-vasp-template-cancel", "n_clicks"),
    Input("vasp-template-select", "value"),
    Input("btn-vasp-template-reset", "n_clicks"),
    Input("btn-vasp-template-save", "n_clicks"),
    State("vasp-template-modal", "is_open"),
    prevent_initial_call=True
)
def handle_vasp_template_modal(n_open, n_cancel, selected_template, n_reset, n_save, is_open):
    trigger = ctx.triggered_id
    
    if trigger == "btn-edit-vasp-template":
        # 打开对话框，加载第一个模板
        content = SESSION_EDITOR.get_vasp_template("geometry_opt")
        return True, content
    
    elif trigger == "btn-vasp-template-cancel" or trigger == "btn-vasp-template-save":
        # 关闭对话框
        return False, no_update
    
    elif trigger == "vasp-template-select" and is_open:
        # 用户选择了不同的模板
        content = SESSION_EDITOR.get_vasp_template(selected_template)
        return True, content
    
    elif trigger == "btn-vasp-template-reset":
        # 重置为默认模板
        default_content = VaspCalculationTemplates.get_template(selected_template)
        return True, default_content
    
    return is_open, no_update


# 合并保存模板的回调 (VASP 和 Queue)
@app.callback(
    Output("template-editor-message", "data"),
    Input("btn-vasp-template-save", "n_clicks"),
    Input("btn-queue-template-save", "n_clicks"),
    State("vasp-template-select", "value"),
    State("vasp-template-textarea", "value"),
    State("queue-template-select", "value"),
    State("queue-template-textarea", "value"),
    prevent_initial_call=True
)
def save_template(n_save_vasp, n_save_queue, vasp_template_name, vasp_content, queue_system, queue_content):
    trigger = ctx.triggered_id
    
    if trigger == "btn-vasp-template-save":
        if not n_save_vasp: raise PreventUpdate
        success, msg = SESSION_EDITOR.update_vasp_template(vasp_template_name, vasp_content)
        return msg
        
    elif trigger == "btn-queue-template-save":
        if not n_save_queue: raise PreventUpdate
        success, msg = SESSION_EDITOR.update_queue_template(queue_system, queue_content)
        return msg
        
    raise PreventUpdate


# 打开、选择和重置队列模板编辑对话框（合并回调以避免重复输出）
@app.callback(
    Output("queue-template-modal", "is_open"),
    Output("queue-template-textarea", "value"),
    Input("btn-edit-queue-template", "n_clicks"),
    Input("btn-queue-template-cancel", "n_clicks"),
    Input("queue-template-select", "value"),
    Input("btn-queue-template-reset", "n_clicks"),
    Input("btn-queue-template-save", "n_clicks"),
    State("queue-template-modal", "is_open"),
    prevent_initial_call=True
)
def handle_queue_template_modal(n_open, n_cancel, selected_queue, n_reset, n_save, is_open):
    trigger = ctx.triggered_id
    
    if trigger == "btn-edit-queue-template":
        # 打开对话框，加载第一个模板
        content = SESSION_EDITOR.get_queue_template("slurm")
        return True, content
    
    elif trigger == "btn-queue-template-cancel" or trigger == "btn-queue-template-save":
        # 关闭对话框
        return False, no_update
    
    elif trigger == "queue-template-select" and is_open:
        # 用户选择了不同的队列系统
        content = SESSION_EDITOR.get_queue_template(selected_queue)
        return True, content
    
    elif trigger == "btn-queue-template-reset":
        # 重置为默认模板
        default_content = QueueSystemTemplates.get_template(selected_queue)
        return True, default_content
    
    return is_open, no_update





if __name__ == '__main__':
    # Dash 入口，固定端口 8051，避免与其他模块冲突
    app.run(debug=True, port=8051)