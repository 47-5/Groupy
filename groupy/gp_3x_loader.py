import pandas as pd
import os


class Loader:
    def __init__(self, ):
        basepath = os.path.abspath(__file__)
        folder_path1 = os.path.dirname(basepath)
        folder_path2 = os.path.join(folder_path1, '..', '..', '..', 'groupy_internal_data')
        folder_path3 = os.path.join(folder_path1, '..', '..', '..', '..', 'groupy_internal_data')

        parameters_filename = os.path.join('group_contribution_parameters.xlsx')
        group_order_filename = os.path.join('group_order.xlsx')
        self.parameters_path = [
            os.path.join(folder_path1, parameters_filename),
            os.path.join(folder_path2, parameters_filename),
            os.path.join(folder_path3, parameters_filename),
                                ]
        self.group_order_path = [
            os.path.join(folder_path1, group_order_filename),
            os.path.join(folder_path2, group_order_filename),
            os.path.join(folder_path3, group_order_filename)
            ]

    def load_parameters(self, parameter_type='simultaneous', split=False):
        assert parameter_type in ['simultaneous', 'step_wise'], '请确保参数类型为simultaneous或step_wise!'
        for path in self.parameters_path:
            try:
                step_wise_first_order = pd.read_excel(path, sheet_name='{}_first_order'.format(parameter_type), index_col='index').T.to_dict()
                step_wise_second_order = pd.read_excel(path, sheet_name='{}_second_order'.format(parameter_type), index_col='index').T.to_dict()
                step_wise_third_order = pd.read_excel(path, sheet_name='{}_third_order'.format(parameter_type), index_col='index').T.to_dict()
                step_wise_universal_constants = pd.read_excel(path, sheet_name='{}_constants'.format(parameter_type), index_col='index').T.to_dict()
                break
            except:
                pass
        else:
            raise FileNotFoundError(f'Can not find group_contribution_parameters.xlsx in {self.parameters_path}')


        if split:
            return step_wise_first_order, step_wise_second_order, step_wise_third_order, step_wise_universal_constants
        else:
            return {**step_wise_first_order, **step_wise_second_order, **step_wise_third_order, **step_wise_universal_constants}

    def load_group_order(self):
        for path in self.group_order_path:
            try:
                f_order_group_function_order = (
                        pd.read_excel(path, sheet_name='f')[
                            'index'] - 1).tolist()  # 减1是为了基团序号和列表索引对上,因为python列表里的索引是从0开始的
                s_order_group_function_order = (
                        pd.read_excel(path, sheet_name='s')['index'] - 1).tolist()  # 减1是为了基团序号和列表索引对上
                t_order_group_function_order = (
                        pd.read_excel(path, sheet_name='t')['index'] - 1).tolist()  # 减1是为了基团序号和列表索引对上
                break
            except:
                pass
        else:
            raise FileNotFoundError(f'Can not find group_order.xlsx in {self.parameters_path}')
        return f_order_group_function_order, s_order_group_function_order, t_order_group_function_order


# if __name__ == '__main__':
#
#     # debug
#     loader = Loader()
#
#     # d = loader.load_parameters()
#     # for i in d:
#     #     print(i)
#     #     print(d[i])

