from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Type

import numpy as np


class DataBatch:
    def __init__(
        self,
        dose: Optional[np.ndarray] = None,
        predicted_dose: Optional[np.ndarray] = None,
        ct: Optional[np.ndarray] = None,
        structure_masks: Optional[np.ndarray] = None,
        structure_mask_names: Optional[List[str]] = None,
        possible_dose_mask: Optional[np.ndarray] = None,
        voxel_dimensions: Optional[np.ndarray] = None,
        patient_list: Optional[List[str]] = None,
        patient_path_list: Optional[List[Path]] = None,
    ):
        self.dose = dose
        self.predicted_dose = predicted_dose
        self.ct = ct
        self.structure_masks = structure_masks
        self.structure_mask_names = structure_mask_names
        self.possible_dose_mask = possible_dose_mask
        self.voxel_dimensions = voxel_dimensions
        self.patient_list = patient_list
        self.patient_path = patient_path_list

    @classmethod
    def initialize_from_required_data(cls, data_dimensions: Dict[str, np.ndarray], batch_size: int) -> DataBatch:
        attribute_values = {}
        for data, dimensions in data_dimensions.items():
            batch_data_dimensions = (batch_size, *dimensions)
            attribute_values[data] = np.zeros(batch_data_dimensions)
        return cls(**attribute_values)

    def set_values(self, data_name: str, batch_index: int, values: np.ndarray):
        getattr(self, data_name)[batch_index] = values

    def get_index_structure_from_structure(self, structure_name: str):
        return self.structure_mask_names.index(structure_name)
