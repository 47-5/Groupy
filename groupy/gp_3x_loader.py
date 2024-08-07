import pandas as pd
import os


class Loader:
    def __init__(self, load_path=os.path.join('gp_3x_internal_data', 'group_contribution_parameters.xlsx')):
        self.load_path = load_path

    def load_parameters(self, parameter_type='simultaneous', split=False):
        assert parameter_type in ['simultaneous', 'step_wise'], '请确保参数类型为simultaneous或step_wise!'
        step_wise_first_order = pd.read_excel(self.load_path, sheet_name='{}_first_order'.format(parameter_type), index_col='index').T.to_dict()
        step_wise_second_order = pd.read_excel(self.load_path, sheet_name='{}_second_order'.format(parameter_type), index_col='index').T.to_dict()
        step_wise_third_order = pd.read_excel(self.load_path, sheet_name='{}_third_order'.format(parameter_type), index_col='index').T.to_dict()
        step_wise_universal_constants = pd.read_excel(self.load_path, sheet_name='{}_constants'.format(parameter_type), index_col='index').T.to_dict()

        if split:
            return step_wise_first_order, step_wise_second_order, step_wise_third_order, step_wise_universal_constants
        else:
            return {**step_wise_first_order, **step_wise_second_order, **step_wise_third_order, **step_wise_universal_constants}


if __name__ == '__main__':

    # debug
    loader = Loader()

    d = loader.load_parameters()
    for i in d:
        print(i)
        print(d[i])

