import pandas as pd


class Tool:
    def __init__(self):
        pass

    def __repr__(self):
        return 'This is a object implemented some useful tools'

    @staticmethod
    def load_smiles_iterator(smiles_file_path):
        print('reading input file...')
        if smiles_file_path.endswith('.txt'):
            smiles_iterator = list(open(smiles_file_path))
        elif smiles_file_path.endswith('.xlsx'):
            smiles_iterator = pd.read_excel(smiles_file_path)['smiles']
        elif smiles_file_path.endswith('.csv'):
            smiles_iterator = pd.read_csv(smiles_file_path)['smiles']
        else:
            raise NotImplemented('无法识别的文件类型，请以.txt/.xlsx/.csv类型的文件作为输入。')
        smiles_iterator = [i.strip() for i in smiles_iterator]
        return smiles_iterator


def export_a_dict(result_dict, export_path='result.csv'):
    df = pd.DataFrame([result_dict])
    df.to_csv(export_path, index_label='index')
    return None


logo = \
'''
---------------------------------------------------------------------------------
Groupy -- A Useful Tool for Molecular Analysis 
Developer: Ruichen Liu
Hint: Please feel easy to contact the developer if you have any problems in use.
E-mail1: liuruichen@tju.edu.cn
E-mail2: 1197748182@qq.com (may reply more quickly than E-mail1)
---------------------------------------------------------------------------------
'''


# if __name__ == '__main__':
#     print('debug gp_3x_tool.py')
#
#     t = Tool()
#
#     t.smi_to_xyz('C1CCCC1')
#     # t.batch_smi_to_xyz(smiles_file_path=r'gp_3x_test_mol\SMILES.txt', xyz_root_path='test_xyz')
#     # t.convert_file_type(in_format='xyz', in_path='C1CCCC1.xyz', out_format='mol2')
#     # t.batch_convert_file_type(in_format='xyz', in_root_path='test_xyz', out_format='mol2', out_root_path=None)
#     # t.batch_convert_file_type(in_format='xyz', in_root_path='test_xyz', out_format='mol2', out_root_path='test_mol2')
#
#     t.smi_to_gjf(smi='C1CCC1', add_other_std_tasks=True)
#     t.batch_smi_to_gjf(smiles_file_path=r'gp_3x_test_mol\SMILES.txt', gjf_root_path='test_gjf', add_other_std_tasks=True)
