# N2RR, NO3-RR, NO2-RR, NORR, CO2RR, ORR, OER
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional
color_list = ['#C22F2F', '#449945', '#1F70A9', '#EA7827',  '#6A5ACD', '#DAA520', '#F7B7D2', '#83639F']
#                 红    ,    绿    ,    蓝    ,    橙     ,     紫    ,     金    ,   粉红   ,    深紫
# 分子能量计算
small_molecular = {
    "h": -3.409,
    "1/2H2(g)": -3.409,
    "H2(g)": -6.8185,
    "O2(g)": -9.910,
    "H2O(l)": -14.220,
    "N2(g)": -17.44,  # DFT:-17.08
    "HNO3": -28.730,
    "NH3(g)": -19.541186,  # E0 -.19541186E+02, -.19541186E+02, -.19542010E+02
    "H2O2(l)": -18.080,
    "NO2(g)": -16.800,
    "HNO2(g)": -22.13,  # -22.03(热化学数据) 势垒加0.2 #-21.85
    "NO(g)": -12.313042,
}
# A + B -
#NO2RR = {
#    "path1": [
#        (["*", "HNO2(g)"], ["NO2", "1/2H2(g)"]),  # 就是修正下气态, 同理我也能全算给NO2-去修正, 就是给HNO2(g) + 0.14, 姑且算0.2
#        (["NO2", "h"], ["NO2H"]),
#        (["NO2H", "h"], ["NO", "H2O(l)"])]

import re
# 从反应方程式得到路径字典,每个反应的第一个会写入字典,最后一步第一个产物写入字典最为标签
def get_label(reaction):
    path_label = {}
    for path, steps in reaction.items():
        path_list = [step[0][0] for step in steps]
        path_list.append(steps[-1][-1][0])
        path_label[path] = path_list
    return path_label

def get_y(file_path, ref, reaction, U=0.0, pH=1):

    df = pd.read_csv(file_path, delim_whitespace=True) # sep='\s+'
    absorbate = dict(zip(df[df.columns[0]], df["G"]))
    file_gibbs_dict = {k: v - absorbate[ref] for k, v in absorbate.items()}
    if absorbate.keys() & small_molecular.keys():
        raise ValueError(f"名称冲突: {absorbate.keys() & small_molecular.keys()}")
    gibbs_dict = {**file_gibbs_dict, **small_molecular}
    gibbs_dict["h"] += -0.0592 * pH - U

    plot_y = {}
    path_metrics = []

    for path, steps in reaction.items():
        plot_y[path] = []
        current_y = gibbs_dict[steps[0][0][0]]
        plot_y[path].append(current_y)

        barrier_steps = []  # 改用列表存储势垒与步骤的元组

        for step_idx, (reactants, products) in enumerate(steps):
            delta_G = sum(gibbs_dict[p] for p in products) - sum(gibbs_dict[r] for r in reactants)

            # 记录势垒值和对应步骤（允许重复）
            barrier_steps.append((delta_G, f"Step {step_idx + 1}: {reactants}→{products}"))
            current_y += delta_G
            plot_y[path].append(current_y)

        formatted_data = [(f"{num:.3f}", desc) for num, desc in barrier_steps]
        #print(f"detailed {path}", formatted_data)
        # 按势垒值降序排序（若值相同，按步骤顺序排列）
        sorted_barriers = sorted(barrier_steps, key=lambda x: (-x[0], x[1]))

        # 提取排序后的势垒序列用于比较
        barrier_values = [item[0] for item in sorted_barriers]
        step_descriptions = [item[1] for item in sorted_barriers]

        path_metrics.append({
            'path': path,
            'barriers': barrier_values,  # 示例: [1.5, 1.5, 0.8]
            'steps': step_descriptions  # 对应的步骤描述
        })

    with open(file_path, 'r') as file:
        first_line = file.readline().strip()
        first_word = first_line.split()[0]  # 获取文件第一行的第一个单词

    best_metric = min(path_metrics, key=lambda x: x['barriers'])

    best = {best_metric['path']: plot_y[best_metric['path']]}  # first_word

    # 打印
    max_barrier = best_metric['barriers'][0]
    #print(best_metric['barriers'])

    rate_determining_steps = [step for barrier, step in zip(best_metric['barriers'], best_metric['steps']) if barrier == max_barrier]
    # print(f"最佳路径: {best_metric['path']}")
    print("决速步:", ", ".join(rate_determining_steps))
    print(f"势垒值: {max_barrier:.2f} eV")  # 假设单位为eV，保留两位小数

    return plot_y, best



# 如果三条路径, 两个重名, 不同名字卡在中间, 画图可能不合理. 应该也没有这种情况
def get_unique_label(path_label, plot_y, keep_min_y=True):
    labels = []
    # 遍历所有路径的中间体和能量
    for path, steps in path_label.items():

        #print(plot_y)
        energies = plot_y[path]
        for i, step in enumerate(steps):
            rounded_energy = round(energies[i], 5)
            labels.append([step, i, rounded_energy])

    # 基础去重（保留完全相同的项中的一个）
    unique_labels = list(map(list, set(map(tuple, labels))))
    # 若启用"同x同标签保留最小y"逻辑
    if keep_min_y:
        min_entries = {}
        for entry in unique_labels:
            step_name, i, energy = entry
            key = (i, step_name)
            if key not in min_entries or energy < min_entries[key][2]:
                min_entries[key] = entry
        unique_labels = list(min_entries.values())

    return unique_labels

def plot_label(ax, labels):
    # 按横坐标分组
    group_by_x = {}
    for i in labels:
        label, x, y = i  # 名称, x , y
        if label not in list(small_molecular.keys()) and label != '*':  # label不为*
            label += '$^*$'  # 使用LaTeX语法表示上标*
        pattern = r'([a-zA-Z])(\d+)'
        label = re.sub(pattern, r'\1$_{\2}$', label)
        # label = re.sub(r'(\d+)', r'$_\1$', label)

        if x not in group_by_x:
            group_by_x[x] = []
        group_by_x[x].append((label, y))
    # 按 x 分别处理
    for x, labels in group_by_x.items():
        # 按 y 值从大到小排序
        labels = sorted(labels, key=lambda item: item[1], reverse=True)
        # 计算偏移
        num = len(labels)
        if num == 1:
            label, y = labels[0]
            ax.text(x * 2 + 0.5, y-0.07, label, fontsize=14, ha='center', va='top')  #1 - 0.15
        else:
            for i, (label, y) in enumerate(labels):
                if i == 0:
                    adjusted_y = y + 0.3   # 控制最高的 0.3
                elif i == num - 1:
                    adjusted_y = y - 0.3  # 控制最低的 0.3
                else:
                    pre_y = labels[i - 1][1]
                    next_y = labels[i + 1][1]
                    if (y - next_y) >= (pre_y - y):
                        adjusted_y = y + 0.4  #0.3
                    else:
                        adjusted_y = y + 0.22  #0.2
                ax.text(x * 2 + 0.5, adjusted_y, label, fontsize=14, ha='center', va='center')

from dataclasses import dataclass
@dataclass
class StepsGraph:
    legend_txt: str = None
    labels: Optional[str] = None # 可以是字符串或None
    def plot_step(self, ax: plt.Axes, y: List[float], color: Optional[str] = None, path_label: str = None):
        x = range(len(y))
        # 创建台阶效果的x和y列表
        y_new = []
        x_new = []
        for j in range(len(y)):
            y_new.append(y[j])
            y_new.append(y[j])
            x_new.append(2 * j)
            x_new.append(2 * j + 1)
        # 绘制虚线连接点
        ax.plot(x_new, y_new, linestyle='--', linewidth=2, color=color, dashes=[1, 1])
        # 绘制每段的实线部分
        k = 0
        while k < len(y_new):
            x_line = [x_new[k], x_new[k + 1]]
            y_line = [y_new[k], y_new[k + 1]]
            ax.plot(x_line, y_line, linestyle='-', linewidth=4, color=color, label=path_label if k == 0 else None, solid_capstyle='round')
            k += 2

from itertools import cycle

def step_graph(filenames, reaction, molecule, figsize=(10, 6), ylim=(-10, 5), dpi=300, U=0.0, pH=0):

    colors = cycle(color_list)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    if isinstance(filenames, str):
        paths_dict = get_label(reaction)

        energy_dict, best = get_y(filenames, molecule, reaction, U, pH)

        label_dict = get_unique_label(paths_dict, energy_dict)
        # print('label_dict', label_list)
        for path_label, gibbs_values in energy_dict.items():

            color = next(colors)
            graph = StepsGraph(legend_txt=path_label)
            graph.plot_step(ax, gibbs_values, color=color, path_label=path_label)

        with open(filenames, 'r') as file:
            first_line = file.readline().strip()
            first_word = first_line.split()[0]  # 获取文件第一行的第一个单词

    else:
        paths_dict_all = {}
        best_dict_all = {}

        # 这是针对最佳路径的, 但是单一路径情况下就是最佳. 不同基底的两条路径就不使用了. 不能考虑副反应
        # 每一个的两条路径(一条是比如HER是有可能的)
        # 可以添加不同基底的多条路径.

        for filename in filenames:
            # 获取数据
            plot_y, plot_y_best = get_y(filename, molecule, reaction, U, pH)
            best_dict_all.update(plot_y_best)
            # 获取路径标签
            paths_dict = get_label(reaction)
            # 提取文件名（无扩展名）
            # f = os.path.splitext(os.path.basename(filename))[0]  # 合并两步操作为一行

            with open(filename, 'r') as file:
                first_line = file.readline().strip()
                f = first_line.split()[0]  # 获取文件第一行的第一个单词

            # 提取最佳路径的键和值

            best_key = next(iter(plot_y_best))  # 更高效的方式获取第一个key
            best_value = plot_y_best[best_key]
            best_dict_all[f] = best_value  # 等价于原逻辑但更简洁
            paths_dict_all[f] = paths_dict[best_key]

            # 为每条路径绘制台阶图
            for path_label, gibbs_values in plot_y_best.items():  # energy_dict

                color = next(colors)
                path_label = f
                graph = StepsGraph(legend_txt=path_label)
                graph.plot_step(ax, gibbs_values, color=color, path_label=path_label)

        label_dict = get_unique_label(paths_dict_all, best_dict_all)

        # 最终自定义区域
        # label自定义
        # label_dict.append(['NO2', 2, -2.12])

        var_name = lambda var: [n for n in globals() if globals()[n] is var][0]
        first_word = var_name(reaction)

    #ax.text(0.97, 0.97, first_word, transform=ax.transAxes, fontsize=25, ha='right', va='top', fontweight='bold')
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_xlabel("Reaction Pathway", fontweight='bold')
    ax.set_ylabel("Gibbs Energy (eV)", fontweight='bold')
    ax.set_ylim(ylim)
    plot_label(ax, label_dict)

    from matplotlib.ticker import MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(0.5))  # Y轴主刻度间距为1
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))  # Y轴副刻度间距为0.5
    #ax.legend(loc='lower left', frameon=False, bbox_to_anchor=(0.02, 0), fontsize=20)  # , prop={'weight': 'bold'}

    handles, labels = ax.get_legend_handles_labels()

    # 分割为前3条和后3条
    handles_left = handles[:4]
    labels_left = labels[:4]
    handles_right = handles[4:6]
    labels_right = labels[4:6]

    # 创建左下角图例
    leg_left = ax.legend(
        handles_left, labels_left,
        loc='lower left',
        frameon=False,
        bbox_to_anchor=(0.02, 0),
        fontsize=14
    )

    # 创建右下角图例
    leg_right = ax.legend(
        handles_right, labels_right,
        loc='lower right',
        frameon=False,
        bbox_to_anchor=(1.02, 0),
        fontsize=14
    )

    ax.add_artist(leg_left)
    ax.add_artist(leg_right)


    plt.tight_layout()
    plt.savefig(f'{filenames}_U_{U}_pH_{pH}.png', bbox_inches='tight')
    plt.show()

NORR = {
    "path1": [
        (["*", "NO(g)"], ["NO"]),
        (["NO", "h"], ["NOH"]),
        (["NOH", "h"], ["N", "H2O(l)"]),
        (["N", "h"], ["NH"]),
        (["NH", "h"], ["NH2"]),
        (["NH2", "h"], ["NH3(g)", "*"]),
    ],
    "path2": [
        (["*", "NO(g)"], ["NO"]),
        (["NO", "h"], ["NOH"]),
        (["NOH", "h"], ["NHOH"]),
        (["NHOH", "h"], ["NH", "H2O(l)"]),
        (["NH", "h"], ["NH2"]),
        (["NH2", "h"], ["NH3(g)", "*"]),
    ],
    "path3": [
        (["*", "NO(g)"], ["NO"]),
        (["NO", "h"], ["HNO"]),
        (["HNO", "h"], ["NH2O"]),
        (["NH2O", "h"], ["NH", "H2O(l)"]),
        (["NH", "h"], ["NH2"]),
        (["NH2", "h"], ["NH3(g)", "*"]),
    ],
    "path4": [
        (["*", "NO(g)"], ["NO"]),
        (["NO", "h"], ["HNO"]),
        (["HNO", "h"], ["NHOH"]),
        (["NHOH", "h"], ["NH", "H2O(l)"]),
        (["NH", "h"], ["NH2"]),
        (["NH2", "h"], ["NH3(g)", "*"]),
    ]
}

NORR1 = {
    "path1": [
        (["*", "NO(g)"], ["NO"]),
        (["NO", "h"], ["NOH"]),
        (["NOH", "h"], ["N", "H2O(l)"]),
        (["N", "h"], ["NH"]),
        (["NH", "h"], ["NH2"]),
        (["NH2", "h"], ["NH3"]),
        (["NH3"], ["*", "NH3(g)"]),
        #(["NH2", "h"], ["NH3(g)", "*"]),
    ],
    "path2": [
        (["*", "NO(g)"], ["NO"]),
        (["NO", "h"], ["NOH"]),
        (["NOH", "h"], ["NHOH"]),
        (["NHOH", "h"], ["NH", "H2O(l)"]),
        (["NH", "h"], ["NH2"]),
        (["NH2", "h"], ["NH3"]),
        (["NH3"], ["*", "NH3(g)"]),
    ],
    "path3": [
        (["*", "NO(g)"], ["NO"]),
        (["NO", "h"], ["HNO"]),
        (["HNO", "h"], ["NH2O"]),
        (["NH2O", "h"], ["NH", "H2O(l)"]),
        (["NH", "h"], ["NH2"]),
        (["NH2", "h"], ["NH3"]),
        (["NH3"], ["*", "NH3(g)"]),
    ],
    "path4": [
        (["*", "NO(g)"], ["NO"]),
        (["NO", "h"], ["HNO"]),
        (["HNO", "h"], ["NHOH"]),
        (["NHOH", "h"], ["NH", "H2O(l)"]),
        (["NH", "h"], ["NH2"]),
        (["NH2", "h"], ["NH3"]),
        (["NH3"], ["*", "NH3(g)"]),
    ]
}

# 自定义反应路径
NO2RR = {
    "path1": [
        (["*", "HNO2(g)"], ["NO2", "1/2H2(g)"]),  # 就是修正下气态, 同理我也能全算给NO2-去修正, 就是给HNO2(g) + 0.14, 姑且算0.2
        (["NO2", "h"], ["NO2H"]),
        (["NO2H", "h"], ["NO", "H2O(l)"]),
        (["NO", "h"], ["NOH"]),
        (["NOH", "h"], ["N", "H2O(l)"]),
        (["N", "h"], ["NH"]),
        (["NH", "h"], ["NH2"]),
        (["NH2", "h"], ["NH3(g)", "*"]),
    ],
    "path2": [
        (["*", "HNO2(g)"], ["NO2", "1/2H2(g)"]),
        #(["*", "NO2(g)"], ["NO2"]),
        (["NO2", "h"], ["NO2H"]),
        (["NO2H", "h"], ["NO", "H2O(l)"]),
        (["NO", "h"], ["NOH"]),
        (["NOH", "h"], ["NHOH"]),
        (["NHOH", "h"], ["NH", "H2O(l)"]),
        (["NH", "h"], ["NH2"]),
        (["NH2", "h"], ["NH3(g)", "*"]),
    ],
    "path3": [
        (["*", "HNO2(g)"], ["NO2", "1/2H2(g)"]),
        #(["*", "NO2(g)"], ["NO2"]),
        (["NO2", "h"], ["NO2H"]),
        (["NO2H", "h"], ["NO", "H2O(l)"]),
        (["NO", "h"], ["HNO"]),
        (["HNO", "h"], ["NH2O"]),
        (["NH2O", "h"], ["NH", "H2O(l)"]),
        (["NH", "h"], ["NH2"]),
        (["NH2", "h"], ["NH3(g)", "*"]),
    ],
    "path4": [
        (["*", "HNO2(g)"], ["NO2", "1/2H2(g)"]),
        #(["*", "NO2(g)"], ["NO2"]),
        (["NO2", "h"], ["NO2H"]),
        (["NO2H", "h"], ["NO", "H2O(l)"]),
        (["NO", "h"], ["HNO"]),
        (["HNO", "h"], ["NHOH"]),
        (["NHOH", "h"], ["NH", "H2O(l)"]),
        (["NH", "h"], ["NH2"]),
        (["NH2", "h"], ["NH3(g)", "*"]),
    ]
}

NO2RR1 = {
    "path1": [
        (["*", "HNO2(g)"], ["NO2", "1/2H2(g)"]),
        (["NO2", "h"], ["NO2H"]),
        (["NO2H", "h"], ["NO", "H2O(l)"]),
        (["NO", "h"], ["NOH"]),
        (["NOH", "h"], ["N", "H2O(l)"]),
        (["N", "h"], ["NH"]),
        (["NH", "h"], ["NH2"]),
        (["NH2", "h"], ["NH3"]),
        (["NH3"], ["NH3(g)", "*"]),
    ],
    "path2": [
        (["*", "HNO2(g)"], ["NO2", "1/2H2(g)"]),
        (["NO2", "h"], ["NO2H"]),
        (["NO2H", "h"], ["NO", "H2O(l)"]),
        (["NO", "h"], ["NOH"]),
        (["NOH", "h"], ["NHOH"]),
        (["NHOH", "h"], ["NH", "H2O(l)"]),
        (["NH", "h"], ["NH2"]),
        (["NH2", "h"], ["NH3"]),
        (["NH3"], ["NH3(g)", "*"]),
    ],
    "path3": [
        (["*", "HNO2(g)"], ["NO2", "1/2H2(g)"]),
        (["NO2", "h"], ["NO2H"]),
        (["NO2H", "h"], ["NO", "H2O(l)"]),
        (["NO", "h"], ["HNO"]),
        (["HNO", "h"], ["NH2O"]),
        (["NH2O", "h"], ["NH", "H2O(l)"]),
        (["NH", "h"], ["NH2"]),
        (["NH2", "h"], ["NH3"]),
        (["NH3"], ["NH3(g)", "*"]),
    ],
    "path4": [
        (["*", "HNO2(g)"], ["NO2", "1/2H2(g)"]),
        (["NO2", "h"], ["NO2H"]),
        (["NO2H", "h"], ["NO", "H2O(l)"]),
        (["NO", "h"], ["HNO"]),
        (["HNO", "h"], ["NHOH"]),
        (["NHOH", "h"], ["NH", "H2O(l)"]),
        (["NH", "h"], ["NH2"]),
        (["NH2", "h"], ["NH3"]),
        (["NH3"], ["NH3(g)", "*"]),
    ]
}

HER ={
    "path1": [
    [["*", "h"], ["H"]],
    [["H"], ["1/2H2(g)", "*"]]
    ]
}

OR = {
    "path1": [
        (["OH", "h"], ["*", "H2O(l)"])
    ]
}

ORR = {
    "path1": [
        (['*', 'NO2-'], ['NO2']),
        (["NO2", "h"], ["NO2H"]),
        (["NO2H", "h"], ["NO", "H2O(l)"]),
        (["NO", "h"], ["NOH"]),
        (["NOH", "h"], ["N", "H2O(l)"]),
        (["N", "h"], ["NH"]),
        (["NH", "h"], ["NH2"]),
        (["NH2", "h"], ["NH3"]),
        (["NH3"], ["NH3(g)", "*"])],
    "path4": [
        (['*', 'NO2-'], ['NO2']),
        (["NO2", "h"], ["NO2H"]),
        (["NO2H", "h"], ["NO", "H2O(l)"]),
        (["NO", "h"], ["HNO"]),
        (["HNO", "h"], ["NHOH"]),
        (["NHOH", "h"], ["NH", "H2O(l)"]),
        (["NH", "h"], ["NH2"]),
        (["NH2", "h"], ["NH3"]),
        (["NH3"], ["NH3(g)", "*"]),
    ]
}

NO2RR2 = {
    "path1": [
        (["*", "HNO2(g)"], ["NO2", "1/2H2(g)"]),  # 就是修正下气态, 同理我也能全算给NO2-去修正, 就是给HNO2(g) + 0.14, 姑且算0.2
        (["NO2", "h"], ["NO2H"]),
        (["NO2H", "h"], ["NO", "H2O(l)"]),
        (["NO", "h"], ["NOH"]),
        (["NOH", "h"], ["N", "H2O(l)"]),
        (["N", "h"], ["NH"]),
        (["NH", "h"], ["NH2"]),
        (["NH2", "h"], ["NH3(g)", "*"]),

    ],
    "path2": [
        (["*", "HNO2(g)"], ["NO2", "1/2H2(g)"]),
        #(["*", "NO2(g)"], ["NO2"]),
        (["NO2", "h"], ["NO2H"]),
        (["NO2H", "h"], ["NO", "H2O(l)"]),
        (["NO", "h"], ["NOH"]),
        (["NOH", "h"], ["NHOH"]),
        (["NHOH", "h"], ["NH", "H2O(l)"]),
        (["NH", "h"], ["NH2"]),
        (["NH2", "h"], ["NH3(g)", "*"]),
    ],
    "path3": [
        (["*", "HNO2(g)"], ["NO2", "1/2H2(g)"]),
        #(["*", "NO2(g)"], ["NO2"]),
        (["NO2", "h"], ["NO2H"]),
        (["NO2H", "h"], ["NO", "H2O(l)"]),
        (["NO", "h"], ["HNO"]),
        (["HNO", "h"], ["NH2O"]),
        (["NH2O", "h"], ["NH", "H2O(l)"]),
        (["NH", "h"], ["NH2"]),
        (["NH2", "h"], ["NH3(g)", "*"]),
    ],
    "path4": [
        (["*", "HNO2(g)"], ["NO2", "1/2H2(g)"]),
        #(["*", "NO2(g)"], ["NO2"]),
        (["NO2", "h"], ["NO2H"]),
        (["NO2H", "h"], ["NO", "H2O(l)"]),
        (["NO", "h"], ["HNO"]),
        (["HNO", "h"], ["NHOH"]),
        (["NHOH", "h"], ["NH", "H2O(l)"]),
        (["NH", "h"], ["NH2"]),
        (["NH2", "h"], ["NH3(g)", "*"]),
    ]
}

OER = {
    "path1": [
        (['*', 'H2O(l)'], ['OH', 'h']),
        (["OH"], ["O", "h"]),
        (["O", "H2O(l)"], ["OOH", "h"]),
        (["OOH"], ["O2(g)", "*", "h"])]
    }

CO2RR = {
    "path1": [
    #(['Cu', 'CO(g)'], ["CO"]),
    (['CO', 'h'], ["HCO"]),
    (['HCO', 'h'], ["CH2O"]),
    (['CH2O', 'h'], ["CH3O"]),
    (['CH3O', 'h'], ["CH3OH"])]
}