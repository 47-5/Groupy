from copy import deepcopy
from functools import lru_cache
from importlib.resources import as_file, files

import pandas as pd


def _data_paths():
    data_root = files('groupy')
    return {
        'parameters': [data_root / 'group_contribution_parameters.xlsx'],
        'group_order': [data_root / 'group_order.xlsx'],
    }


@lru_cache(maxsize=None)
def _load_parameter_tables(parameter_type):
    parameters_path = _data_paths()['parameters']
    for path in parameters_path:
        if not path.is_file():
            continue
        first_order = Loader._read_excel(
            path,
            sheet_name=f'{parameter_type}_first_order',
            index_col='index',
        ).T.to_dict()
        second_order = Loader._read_excel(
            path,
            sheet_name=f'{parameter_type}_second_order',
            index_col='index',
        ).T.to_dict()
        third_order = Loader._read_excel(
            path,
            sheet_name=f'{parameter_type}_third_order',
            index_col='index',
        ).T.to_dict()
        universal_constants = Loader._read_excel(
            path,
            sheet_name=f'{parameter_type}_constants',
            index_col='index',
        ).T.to_dict()
        return first_order, second_order, third_order, universal_constants
    raise FileNotFoundError(f'Can not find group_contribution_parameters.xlsx in {parameters_path}')


@lru_cache(maxsize=None)
def _load_group_order_tables():
    group_order_path = _data_paths()['group_order']
    for path in group_order_path:
        if not path.is_file():
            continue
        f_order_group_function_order = (
                Loader._read_excel(path, sheet_name='f')[
                    'index'] - 1).tolist()  # 减1是为了基团序号和列表索引对上,因为python列表里的索引是从0开始的
        s_order_group_function_order = (
                Loader._read_excel(path, sheet_name='s')['index'] - 1).tolist()  # 减1是为了基团序号和列表索引对上
        t_order_group_function_order = (
                Loader._read_excel(path, sheet_name='t')['index'] - 1).tolist()  # 减1是为了基团序号和列表索引对上
        return f_order_group_function_order, s_order_group_function_order, t_order_group_function_order
    raise FileNotFoundError(f'Can not find group_order.xlsx in {group_order_path}')


class Loader:
    """
    A class for loading internal date of Groupy. Usually, users do not need to utilize this Python class.
    """
    def __init__(self, ):
        paths = _data_paths()
        self.parameters_path = paths['parameters']
        self.group_order_path = paths['group_order']

    @staticmethod
    def _read_excel(data_file, **kwargs):
        with as_file(data_file) as path:
            return pd.read_excel(path, **kwargs)

    @staticmethod
    def clear_cache():
        _load_parameter_tables.cache_clear()
        _load_group_order_tables.cache_clear()

    def load_parameters(self, parameter_type='simultaneous', split=False):
        """
        Loading parameters of group contribution method for groupy.gp_calculator.Calculator.
        """
        assert parameter_type in ['simultaneous', 'step_wise'], '请确保参数类型为simultaneous或step_wise!'
        parameter_tables = _load_parameter_tables(parameter_type)

        if split:
            return deepcopy(parameter_tables)

        parameters = {}
        for table in parameter_tables:
            parameters.update(deepcopy(table))
        return parameters

    def load_group_order(self):
        """
        Loading order of group for groupy.gp_counter.Counter
        """
        return deepcopy(_load_group_order_tables())


# if __name__ == '__main__':
#
#     # debug
#     loader = Loader()
#
#     # d = loader.load_parameters()
#     # for i in d:
#     #     print(i)
#     #     print(d[i])
