from typing import Dict, List


class DataShapes:
    def __init__(self, num_rois):
        self.num_rois = num_rois
        self.patient_shape = (128, 128, 128)

    @property
    def dose(self):
        """Dose deposited within the patient tensor"""
        return self.patient_shape + (1,)

    @property
    def predicted_dose(self):
        """Predicted dose that should be deposited within the patient tensor"""
        return self.dose

    @property
    def ct(self):
        """CT image grey scale within the patient tensor"""
        return self.patient_shape + (1,)

    @property
    def structure_masks(self):
        """Mask of all structures in patient"""
        return self.patient_shape + (self.num_rois,)

    @property
    def possible_dose_mask(self):
        """Mask where dose can be deposited"""
        return self.patient_shape + (1,)

    @property
    def voxel_dimensions(self):
        """Physical dimensions of patient voxels (in mm)"""
        return tuple((3,))

    def from_data_names(self, data_names: List[str]) -> Dict:
        data_shapes = {}
        for name in data_names:
            data_shapes[name] = getattr(self, name)
        return data_shapes
