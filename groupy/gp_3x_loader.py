import pandas as pd
import os


class Loader:
    def __init__(self, ):
        basepath = os.path.abspath(__file__)
        folder = os.path.dirname(basepath)
        parameters_filename = os.path.join('group_contribution_parameters.xlsx')
        group_order_filename = os.path.join('group_order.xlsx')
        self.parameters_path = os.path.join(folder, parameters_filename)
        self.group_order_path = os.path.join(folder, group_order_filename)
        print(basepath)

    def load_parameters(self, parameter_type='simultaneous', split=False):
        assert parameter_type in ['simultaneous', 'step_wise'], '请确保参数类型为simultaneous或step_wise!'
        step_wise_first_order = pd.read_excel(self.parameters_path, sheet_name='{}_first_order'.format(parameter_type), index_col='index').T.to_dict()
        step_wise_second_order = pd.read_excel(self.parameters_path, sheet_name='{}_second_order'.format(parameter_type), index_col='index').T.to_dict()
        step_wise_third_order = pd.read_excel(self.parameters_path, sheet_name='{}_third_order'.format(parameter_type), index_col='index').T.to_dict()
        step_wise_universal_constants = pd.read_excel(self.parameters_path, sheet_name='{}_constants'.format(parameter_type), index_col='index').T.to_dict()

        if split:
            return step_wise_first_order, step_wise_second_order, step_wise_third_order, step_wise_universal_constants
        else:
            return {**step_wise_first_order, **step_wise_second_order, **step_wise_third_order, **step_wise_universal_constants}

    def load_group_order(self):
        f_order_group_function_order = (
                pd.read_excel(self.group_order_path, sheet_name='f')[
                    'index'] - 1).tolist()  # 减1是为了基团序号和列表索引对上,因为python列表里的索引是从0开始的
        s_order_group_function_order = (
                pd.read_excel(self.group_order_path, sheet_name='s')['index'] - 1).tolist()  # 减1是为了基团序号和列表索引对上
        t_order_group_function_order = (
                pd.read_excel(self.group_order_path, sheet_name='t')['index'] - 1).tolist()  # 减1是为了基团序号和列表索引对上
        return f_order_group_function_order, s_order_group_function_order, t_order_group_function_order


if __name__ == '__main__':

    # debug
    loader = Loader()

    # d = loader.load_parameters()
    # for i in d:
    #     print(i)
    #     print(d[i])

