import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
import os

class MLTrainDataBuilder:
    """
    Machine Learning Training Data Builder.
    Handles data preparation for both traditional ML models and GNNs.
    """

    def __init__(self, elements_df: pd.DataFrame):
        """
        Initialize the data builder.

        Args:
            elements_df: DataFrame containing elemental properties.
        """
        self.elements_df = elements_df

    def get_atomic_features(
        self,
        element_symbol: str,
        selected_columns: Optional[List[str]] = None
    ) -> List[float]:
        """
        Get atomic features for a given element.
        """
        row = self.elements_df[self.elements_df['symbol'].str.strip() == element_symbol.strip()]
        if row.empty:
            return []

        if not selected_columns:
            cols = row.select_dtypes(include=['number']).columns.tolist()
            vals = row[cols].values.flatten().tolist()
        else:
            vals = row[selected_columns].values.flatten().tolist()

        return [0.0 if pd.isna(v) else float(v) for v in vals]

    def build_tabular_dataset(
        self,
        structures: List[Dict],
        targets: Dict[str, float],
        atom_indices: List[int],
        selected_columns: Optional[List[str]] = None,
        parse_structure_func: Callable = None
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Build tabular dataset for traditional ML models.

        Args:
            structures: List of structure dicts [{'filename': str, 'content': str}, ...]
            targets: Dictionary of targets {filename: value}
            atom_indices: List of atom indices to extract features from.
            selected_columns: List of elemental property columns to use.
            parse_structure_func: Function to parse structure content.

        Returns:
            (X, y, feature_names)
        """
        if parse_structure_func is None:
            raise ValueError("parse_structure_func is required")

        data = []
        y_values = []
        filenames = []
        
        # Generate feature names
        feature_names = []
        try:
            if structures:
                s0 = parse_structure_func(structures[0]['content'])
                for idx in atom_indices:
                    # Use a dummy element to get column names if needed, or just use the first structure's element
                    # Here we assume all atoms at 'idx' across structures play the same role
                    # We just need the column names from the dataframe
                    if selected_columns:
                        cols = selected_columns
                    else:
                        # Get numeric columns from elements_df
                        cols = self.elements_df.select_dtypes(include=['number']).columns.tolist()
                    
                    feature_names.extend([f"Atom{idx}_{c}" for c in cols])
        except Exception as e:
            print(f"Error generating feature names: {e}")
            # Fallback or re-raise depending on robustness requirements

        for s in structures:
            fname = s['filename']
            # Try matching filename with or without extension
            key = os.path.splitext(fname)[0]
            val = targets.get(fname)
            if val is None:
                val = targets.get(key)

            if val is not None:
                try:
                    struct = parse_structure_func(s['content'])
                    feats = []
                    for idx in atom_indices:
                        if idx < len(struct):
                            # Handle both Pymatgen and ASE objects
                            if hasattr(struct[idx], 'specie'):
                                el = struct[idx].specie.symbol
                            else:
                                el = struct[idx].symbol
                            feats.extend(self.get_atomic_features(el, selected_columns))
                        else:
                            # Handle index out of bounds if necessary
                            pass 
                    
                    if feats:
                        # Check if feature length matches
                        if len(feats) == len(feature_names):
                            data.append(feats)
                            y_values.append(val)
                            filenames.append(fname)
                except Exception as e:
                    print(f"Warning: Failed to process {fname}: {e}")
                    continue

        X = pd.DataFrame(data, columns=feature_names)
        y = pd.Series(y_values, name='target')
        
        return X, y, filenames

    def build_graph_dataset(
        self,
        structures: List[Dict],
        targets: Dict[str, float],
        parse_structure_func: Callable = None
    ) -> List[Dict]:
        """
        Build graph dataset for GNNs.
        
        Returns a list of dicts, each containing:
            - atomic_numbers: List[int]
            - positions: List[List[float]]
            - target: float
            - filename: str
        """
        if parse_structure_func is None:
            raise ValueError("parse_structure_func is required")

        dataset = []
        
        for s in structures:
            fname = s['filename']
            key = os.path.splitext(fname)[0]
            val = targets.get(fname) or targets.get(key)

            if val is not None:
                try:
                    struct = parse_structure_func(s['content'])
                    
                    # Handle both Pymatgen and ASE objects
                    if hasattr(struct, 'get_atomic_numbers'):
                        # ASE
                        atomic_numbers = struct.get_atomic_numbers()
                        positions = struct.get_positions()
                    else:
                        # Pymatgen
                        atomic_numbers = [site.specie.number for site in struct]
                        positions = [site.coords.tolist() for site in struct]
                    
                    dataset.append({
                        'atomic_numbers': np.array(atomic_numbers),
                        'positions': np.array(positions),
                        'target': val,
                        'filename': fname
                    })
                except Exception as e:
                    print(f"Warning: Failed to process {fname} for graph: {e}")
                    continue
                    
        return dataset
