from nnunet.experiment_planning.experiment_planner_baseline_3DUNet_v21 import ExperimentPlanner3D_v21
import numpy as np

class ExperimentPlanner3D_v21_trgSp_z2p2_yx0p9121_baseline(ExperimentPlanner3D_v21):
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super().__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.data_identifier = "nnUNetData_plans_v2.1_trgSp_z2p2_yx0p9121_baseline"
        self.plans_fname = self.data_identifier + "_plans_3D.pkl"
        
        # A100 80GB optimization: Use 60GB for batch size computation
        self.use_this_for_batch_size_computation_3D = 60 * 1024 * 1024 * 1024

    def get_target_spacing(self):
        return np.array([2.2, 0.9121, 0.9121])
