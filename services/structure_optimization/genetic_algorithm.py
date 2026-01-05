"""
遗传算法吸附构型搜索
使用遗传算法优化吸附分子在表面上的位置和方向
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from ase import Atoms
from ase.build import molecule
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import random


class GeneticAdsorptionSearch:
    """
    遗传算法吸附构型搜索
    
    通过遗传算法搜索最优的吸附位点和方向
    """
    
    def __init__(
        self,
        slab: Structure,
        adsorbate_name: str,
        population_size: int = 20,
        n_generations: int = 30,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.7,
        height_range: Tuple[float, float] = (1.5, 3.0),
        calculator=None
    ):
        """
        初始化遗传算法搜索
        
        Args:
            slab: 表面结构(pymatgen Structure对象)
            adsorbate_name: 吸附分子名称(例如 'H2O', 'CO', 'OH')
            population_size: 种群大小
            n_generations: 进化代数
            mutation_rate: 突变概率
            crossover_rate: 交叉概率
            height_range: 吸附高度范围(最小值, 最大值)
            calculator: ASE计算器(如果为None则使用简单的LJ势)
        """
        self.slab_pmg = slab
        self.slab_ase = AseAtomsAdaptor.get_atoms(slab)
        self.adsorbate_name = adsorbate_name
        self.adsorbate_mol = molecule(adsorbate_name)
        
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.height_range = height_range
        self.calculator = calculator
        
        # 获取表面边界
        positions = self.slab_ase.get_positions()
        self.x_range = (positions[:, 0].min(), positions[:, 0].max())
        self.y_range = (positions[:, 1].min(), positions[:, 1].max())
        self.z_max = positions[:, 2].max()
        
        # 固定底层原子
        self.n_layers_fixed = 2
        self._setup_constraints()
        
    def _setup_constraints(self):
        """设置约束固定底层原子"""
        z_coords = self.slab_ase.get_positions()[:, 2]
        z_sorted = np.sort(np.unique(z_coords))
        if len(z_sorted) > self.n_layers_fixed:
            z_threshold = z_sorted[self.n_layers_fixed]
            mask = z_coords < z_threshold
            self.slab_ase.set_constraint(FixAtoms(mask=mask))
    
    def _create_individual(self) -> Dict:
        """
        创建一个个体(一个吸附构型)
        
        Returns:
            个体字典: {'x': x坐标, 'y': y坐标, 'height': 高度, 
                      'theta': 方位角, 'phi': 极角, 'psi': 滚转角}
        """
        return {
            'x': random.uniform(*self.x_range),
            'y': random.uniform(*self.y_range),
            'height': random.uniform(*self.height_range),
            'theta': random.uniform(0, 2*np.pi),  # 方位角
            'phi': random.uniform(0, np.pi),      # 极角
            'psi': random.uniform(0, 2*np.pi)     # 滚转角
        }
    
    def _individual_to_structure(self, individual: Dict) -> Atoms:
        """
        将个体转换为原子结构
        
        Args:
            individual: 个体字典
            
        Returns:
            完整的吸附结构(ASE Atoms对象)
        """
        # 复制底物和吸附物
        slab_copy = self.slab_ase.copy()
        ads_copy = self.adsorbate_mol.copy()
        
        # 旋转吸附物
        ads_copy.euler_rotate(
            phi=individual['phi'],
            theta=individual['theta'],
            psi=individual['psi'],
            center='COM'
        )
        
        # 平移吸附物到指定位置
        ads_com = ads_copy.get_center_of_mass()
        target_pos = np.array([
            individual['x'],
            individual['y'],
            self.z_max + individual['height']
        ])
        ads_copy.translate(target_pos - ads_com)
        
        # 合并结构
        combined = slab_copy + ads_copy
        if self.calculator:
            combined.set_calculator(self.calculator)
            
        return combined
    
    def _evaluate_fitness(self, individual: Dict) -> float:
        """
        评估个体适应度(能量越低越好)
        
        Args:
            individual: 个体字典
            
        Returns:
            适应度值(负能量)
        """
        try:
            structure = self._individual_to_structure(individual)
            
            if self.calculator is None:
                # 使用简化的能量估算(基于距离)
                slab_pos = structure.get_positions()[:len(self.slab_ase)]
                ads_pos = structure.get_positions()[len(self.slab_ase):]
                
                # 计算最近邻距离
                min_dist = np.min([
                    np.linalg.norm(slab_pos - ads_p)
                    for ads_p in ads_pos
                ])
                
                # 简单的LJ势能
                if min_dist < 1.0:
                    energy = 1000.0  # 惩罚太近的构型
                else:
                    energy = 4.0 * ((2.5/min_dist)**12 - (2.5/min_dist)**6)
            else:
                # 使用真实计算器
                try:
                    # 简单单点能量计算
                    energy = structure.get_potential_energy()
                except:
                    # 如果计算失败,进行结构优化
                    opt = BFGS(structure, logfile=None)
                    opt.run(fmax=0.05, steps=50)
                    energy = structure.get_potential_energy()
            
            return -energy  # 负能量作为适应度
            
        except Exception as e:
            print(f"适应度评估失败: {e}")
            return -1e6  # 返回很小的适应度
    
    def _select_parents(self, population: List[Dict], fitnesses: List[float]) -> Tuple[Dict, Dict]:
        """
        锦标赛选择父代
        
        Args:
            population: 种群
            fitnesses: 适应度列表
            
        Returns:
            两个父代个体
        """
        tournament_size = 3
        
        def tournament():
            competitors = random.sample(list(zip(population, fitnesses)), tournament_size)
            return max(competitors, key=lambda x: x[1])[0]
        
        return tournament(), tournament()
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """
        交叉操作
        
        Args:
            parent1, parent2: 父代个体
            
        Returns:
            两个子代个体
        """
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1, child2 = {}, {}
        for key in parent1.keys():
            if random.random() < 0.5:
                child1[key] = parent1[key]
                child2[key] = parent2[key]
            else:
                child1[key] = parent2[key]
                child2[key] = parent1[key]
        
        return child1, child2
    
    def _mutate(self, individual: Dict) -> Dict:
        """
        突变操作
        
        Args:
            individual: 个体
            
        Returns:
            突变后的个体
        """
        mutated = individual.copy()
        
        if random.random() < self.mutation_rate:
            mutated['x'] += random.gauss(0, 0.5)
            mutated['x'] = np.clip(mutated['x'], *self.x_range)
        
        if random.random() < self.mutation_rate:
            mutated['y'] += random.gauss(0, 0.5)
            mutated['y'] = np.clip(mutated['y'], *self.y_range)
        
        if random.random() < self.mutation_rate:
            mutated['height'] += random.gauss(0, 0.2)
            mutated['height'] = np.clip(mutated['height'], *self.height_range)
        
        if random.random() < self.mutation_rate:
            mutated['theta'] += random.gauss(0, 0.3)
            mutated['theta'] = mutated['theta'] % (2*np.pi)
        
        if random.random() < self.mutation_rate:
            mutated['phi'] += random.gauss(0, 0.3)
            mutated['phi'] = np.clip(mutated['phi'], 0, np.pi)
        
        if random.random() < self.mutation_rate:
            mutated['psi'] += random.gauss(0, 0.3)
            mutated['psi'] = mutated['psi'] % (2*np.pi)
        
        return mutated
    
    def search(self, verbose: bool = True) -> List[Tuple[Structure, float]]:
        """
        执行遗传算法搜索
        
        Args:
            verbose: 是否打印进度信息
            
        Returns:
            排序后的结构列表[(Structure, energy), ...]
        """
        # 初始化种群
        population = [self._create_individual() for _ in range(self.population_size)]
        
        best_individuals = []
        
        for generation in range(self.n_generations):
            # 评估适应度
            fitnesses = [self._evaluate_fitness(ind) for ind in population]
            
            # 记录最佳个体
            best_idx = np.argmax(fitnesses)
            best_individual = population[best_idx]
            best_fitness = fitnesses[best_idx]
            
            if verbose:
                print(f"Generation {generation+1}/{self.n_generations}: "
                      f"Best fitness = {best_fitness:.4f}")
            
            # 保存最佳个体
            best_individuals.append((best_individual.copy(), -best_fitness))
            
            # 生成新种群
            new_population = []
            
            # 精英保留
            elite_size = max(2, self.population_size // 10)
            elite_indices = np.argsort(fitnesses)[-elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # 生成剩余个体
            while len(new_population) < self.population_size:
                parent1, parent2 = self._select_parents(population, fitnesses)
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        # 最终评估并排序
        final_structures = []
        seen_energies = set()
        
        for individual, energy in best_individuals:
            # 去重(能量相近的认为是同一构型)
            energy_rounded = round(energy, 3)
            if energy_rounded in seen_energies:
                continue
            seen_energies.add(energy_rounded)
            
            structure_ase = self._individual_to_structure(individual)
            structure_pmg = AseAtomsAdaptor.get_structure(structure_ase)
            final_structures.append((structure_pmg, energy))
        
        # 按能量排序
        final_structures.sort(key=lambda x: x[1])
        
        return final_structures[:10]  # 返回前10个最优构型
