import pandas as pd
from importlib.resources import as_file, files


class Loader:
    """
    A class for loading internal date of Groupy. Usually, users do not need to utilize this Python class.
    """
    def __init__(self, ):
        data_root = files('groupy')
        self.parameters_path = [data_root / 'group_contribution_parameters.xlsx']
        self.group_order_path = [data_root / 'group_order.xlsx']

    @staticmethod
    def _read_excel(data_file, **kwargs):
        with as_file(data_file) as path:
            return pd.read_excel(path, **kwargs)

    def load_parameters(self, parameter_type='simultaneous', split=False):
        """
        Loading parameters of group contribution method for groupy.gp_calculator.Calculator.
        """
        assert parameter_type in ['simultaneous', 'step_wise'], '请确保参数类型为simultaneous或step_wise!'
        for path in self.parameters_path:
            if not path.is_file():
                continue
            step_wise_first_order = self._read_excel(path, sheet_name='{}_first_order'.format(parameter_type), index_col='index').T.to_dict()
            step_wise_second_order = self._read_excel(path, sheet_name='{}_second_order'.format(parameter_type), index_col='index').T.to_dict()
            step_wise_third_order = self._read_excel(path, sheet_name='{}_third_order'.format(parameter_type), index_col='index').T.to_dict()
            step_wise_universal_constants = self._read_excel(path, sheet_name='{}_constants'.format(parameter_type), index_col='index').T.to_dict()
            break
        else:
            raise FileNotFoundError(f'Can not find group_contribution_parameters.xlsx in {self.parameters_path}')


        if split:
            return step_wise_first_order, step_wise_second_order, step_wise_third_order, step_wise_universal_constants
        else:
            return {**step_wise_first_order, **step_wise_second_order, **step_wise_third_order, **step_wise_universal_constants}

    def load_group_order(self):
        """
        Loading order of group for groupy.gp_counter.Counter
        """
        for path in self.group_order_path:
            if not path.is_file():
                continue
            f_order_group_function_order = (
                    self._read_excel(path, sheet_name='f')[
                        'index'] - 1).tolist()  # 减1是为了基团序号和列表索引对上,因为python列表里的索引是从0开始的
            s_order_group_function_order = (
                    self._read_excel(path, sheet_name='s')['index'] - 1).tolist()  # 减1是为了基团序号和列表索引对上
            t_order_group_function_order = (
                    self._read_excel(path, sheet_name='t')['index'] - 1).tolist()  # 减1是为了基团序号和列表索引对上
            break
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
