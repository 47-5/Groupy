from rdkit import Chem
from math import log
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from groupy.gp_loader import Loader
from groupy.gp_counter import Counter


class Calculator:
    """
    A Class for calculating properties of given molecules based on the group contribution method
    基于基团贡献法计算给定分子性质的类
    """
    def __init__(self):
        self.loader = Loader()
        self.counter = Counter()
        self.parameters_step_wise = self.loader.load_parameters(parameter_type='step_wise', split=False)
        self.parameters_simultaneous = self.loader.load_parameters(parameter_type='simultaneous', split=False)

    def __repr__(self):
        return '这是一个使用基团贡献法计算分子性质的计算器'

    @staticmethod
    def Tm(group_number, parameters):
        """
        Calculating the freezing (melting) point. When user call Calculator.calculate_a_mol(), this method will be called automatically.
        In general, users do not need to call this function themselves
        计算冰点（熔点）
        :param group_number: number of different groups in a given molecule.
        In general, users can use the result of groupy.gp_counter.Counter.count_a_mol()
        :param parameters: parameters used in group contribution method.
        In general, users can use internal data of groupy that can be loaded by groupy.gp_loader.Loader
        """
        right_side_eq = 0.00
        for i in group_number:  # 这样写是取了字典的键
            right_side_eq += group_number[i] * parameters[i]['Tm']
        Tm = parameters[1]['Tm0'] * log(max(right_side_eq, 1.0))   # 这里的1.0是为了上计算结果小于0K的都设置为0K
        return round(Tm, 3)

    @staticmethod
    def Tb(group_number, parameters):
        right_side_eq = 0.00
        for i in group_number:
            right_side_eq += group_number[i] * parameters[i]['Tb']
        Tb = parameters[1]['Tb0'] * log(max(right_side_eq, 1.0))  # 这里的1.0是为了上计算结果小于0K的都设置为0K
        return round(Tb, 3)

    @staticmethod
    def Tc(group_number, parameters):
        right_side_eq = 0.00
        for i in group_number:
            right_side_eq += group_number[i] * parameters[i]['Tc']
        Tc = parameters[1]['Tc0'] * log(max(right_side_eq, 1.0))  # 这里的1.0是为了上计算结果小于0K的都设置为0K
        return round(Tc, 3)

    @staticmethod
    def Pc(group_number, parameters):
        right_side_eq = 0.00
        for i in group_number:
            right_side_eq += group_number[i] * parameters[i]['Pc']
        Pc = parameters[1]['Pc1'] + (right_side_eq + parameters[1]['Pc2']) ** -2
        return round(Pc, 4)

    @staticmethod
    def Vc(group_number, parameters):
        right_side_eq = 0.00
        for i in group_number:
            right_side_eq += group_number[i] * parameters[i]['Vc']
        Vc = parameters[1]['Vc0'] + right_side_eq
        return round(Vc, 2)

    @staticmethod
    def delta_Gf(group_number, parameters):
        right_side_eq = 0.00
        for i in group_number:
            right_side_eq += group_number[i] * parameters[i]['Gf']
        delta_Gf = parameters[1]['Gf0'] + right_side_eq
        return round(delta_Gf, 3)

    @staticmethod
    def delta_Hf(group_number, parameters):
        right_side_eq = 0.00
        for i in group_number:
            right_side_eq += group_number[i] * parameters[i]['Hf']
        delta_Hf = parameters[1]['Hf0'] + right_side_eq
        return round(delta_Hf, 3)

    @staticmethod
    def delta_Hv(group_number, parameters):
        right_side_eq = 0.00
        for i in group_number:
            right_side_eq += group_number[i] * parameters[i]['Hv']
        delta_Hv = parameters[1]['Hv0'] + right_side_eq
        return round(delta_Hv, 3)

    @staticmethod
    def delta_Hfus(group_number, parameters):
        right_side_eq = 0.00
        for i in group_number:
            right_side_eq += group_number[i] * parameters[i]['Hfus']
        delta_Hfus = parameters[1]['Hfus0'] + right_side_eq
        return round(delta_Hfus, 3)

    @staticmethod
    def flash_point(group_number, parameters):
        right_side_eq = 0.00
        for i in group_number:
            right_side_eq += group_number[i] * parameters[i]['Fp']
        Fp = parameters[1]['Fp0'] + right_side_eq
        return round(Fp, 3)

    @staticmethod
    def molar_volume(group_number, parameters):
        right_side_eq = 0.00
        for i in group_number:
            right_side_eq += group_number[i] * parameters[i]['Vm']
        Vm = parameters[1]['Vm0'] + right_side_eq
        return round(Vm, 3)

    @staticmethod
    def density(molar_mass, Vs):
        return round(molar_mass / (1000 * Vs), 3)

    @staticmethod
    def delta_Hc(C_number, H_number, delta_Hf):
        delta_Hc = -(-395.51 * C_number - 241.83 * H_number / 2 - delta_Hf)
        return round(delta_Hc, 3)

    @staticmethod
    def q(delta_Hc, molar_mass):
        return round(delta_Hc / molar_mass, 3)

    @staticmethod
    def isp(C_number, H_number, q):
        H_C_ratio = H_number / C_number
        parameter = q * (11.91 + H_C_ratio) / (43.66 + 8.936 * H_C_ratio)
        if parameter < 0:  # todo这是无意中发现的错误，可能是因为算的不是碳氢分子（确实目前发现的出现错误的是含F原子的分子）
            parameter = 0
        isp = (2 * 0.556 * parameter) ** 0.5 / 9.8 * 1000
        return round(isp, 3)

    @staticmethod
    def C_number(mol):
        C_atoms = [i for i in mol.GetAtoms() if i.GetAtomicNum() == 6]
        return len(C_atoms)

    @staticmethod
    def H_number(mol):
        H_atoms = [i for i in Chem.AddHs(mol).GetAtoms() if i.GetAtomicNum() == 1]
        return len(H_atoms)

    @staticmethod
    def smiles(mol):
        return Chem.MolToSmiles(mol)

    @staticmethod
    def molar_mass(mol):
        molar_mass = 0.000
        atoms = Chem.AddHs(mol).GetAtoms()
        for i in atoms:
            molar_mass += i.GetMass()
        return molar_mass

    def calculate_a_mol(self, mol, parameter_type='step_wise', debug=False):
        init_smi = mol
        try:
            if isinstance(mol, str):
                mol = Chem.MolFromSmiles(mol)
            group_number = self.counter.count_a_mol(mol, clear_mode=True, add_note=True)
            if group_number.get('note', ''):
                counter_note = group_number['note']
                del group_number['note']
            else:
                counter_note = ''

            if parameter_type == 'step_wise':
                parameters = self.parameters_step_wise  # todo 这里是速度慢的原因，每算一个分子都要加载一遍参数，要修改
            elif parameter_type == 'simultaneous':
                parameters = self.parameters_simultaneous
            else:
                raise NotImplemented('不可用的参数类型，只能使用step_wise或simultaneous')

            if debug:
                print(group_number)
                # print(parameters)
            Tm = self.Tm(group_number=group_number, parameters=parameters)
            Tb = self.Tb(group_number=group_number, parameters=parameters)
            Tc = self.Tc(group_number=group_number, parameters=parameters)
            Pc = self.Pc(group_number=group_number, parameters=parameters)
            Vc = self.Vc(group_number=group_number, parameters=parameters)
            delta_Gf = self.delta_Gf(group_number=group_number, parameters=parameters)
            delta_Hf = self.delta_Hf(group_number=group_number, parameters=parameters)
            delta_Hv = self.delta_Hv(group_number=group_number, parameters=parameters)
            delta_Hfus = self.delta_Hfus(group_number=group_number, parameters=parameters)
            C_number = self.C_number(mol)
            H_number = self.H_number(mol)
            molar_mass = self.molar_mass(mol)
            flash_point = self.flash_point(group_number=group_number, parameters=parameters)
            molar_volume = self.molar_volume(group_number=group_number, parameters=parameters)
            density = self.density(molar_mass=molar_mass, Vs=molar_volume)
            delta_Hc = self.delta_Hc(C_number=C_number, H_number=H_number, delta_Hf=delta_Hf)
            q = self.q(delta_Hc=delta_Hc, molar_mass=molar_mass)
            isp = self.isp(C_number=C_number, H_number=H_number, q=q)
            smiles = self.smiles(mol)
            return {'smiles': smiles,
                    'molar_mass': molar_mass,
                    'flash_point/K': flash_point,
                    'Tm/K': Tm, 'Tb/K': Tb, 'Tc/K': Tc,
                    'Pc/bar': Pc, 'Vc/(cm3/mol)': Vc,
                    'density/(g/cm3)': density,
                    'delta_G/(KJ/mol)': delta_Gf,
                    'delta_Hf/(KJ/mol)': delta_Hf,
                    'delta_Hvap/(KJ/mol)': delta_Hv,
                    'delta_Hfus/(KJ/mol)': delta_Hfus,
                    'molar_volume/(cm3/mol)(default298K)': molar_volume,
                    'delta_Hc/(KJ/mol)': delta_Hc,
                    'mass_calorific_value_h/(MJ/kg)': q,
                    'ISP': isp,
                    'note': counter_note + ' at 298K'}
        except:
            print(f'Error! There is something wrong when calculating {init_smi}, please check it.')
            return {'smiles': init_smi,
                    'molar_mass': '?',
                    'flash_point/K': '?',
                    'Tm/K': '?', 'Tb/K': '?', 'Tc/K': '?',
                    'Pc/bar': '?', 'Vc/(cm3/mol)': '?',
                    'density/(g/cm3)': '?',
                    'delta_G/(KJ/mol)': '?',
                    'delta_Hf/(KJ/mol)': '?',
                    'delta_Hvap/(KJ/mol)': '?',
                    'delta_Hfus/(KJ/mol)': '?',
                    'molar_volume/(cm3/mol)(default298K)': '?',
                    'delta_Hc/(KJ/mol)': '?',
                    'mass_calorific_value_h/(MJ/kg)': '?',
                    'ISP': '?',
                    'note': 'There must be something wrong with this SMILES'}

    def calculate_mols(self, smiles_file_path, properties_file_path='gp_3x_result.csv', parameter_type='simultaneous'):  # todo 还没实现不同步拟合的参数的使用
        print('reading input file...')
        if smiles_file_path.endswith('.txt'):
            smiles_iterator = list(open(smiles_file_path))
        elif smiles_file_path.endswith('.xlsx'):
            smiles_iterator = pd.read_excel(smiles_file_path)['smiles']
        elif smiles_file_path.endswith('.csv'):
            smiles_iterator = pd.read_csv(smiles_file_path)['smiles']
        else:
            print('无法识别的文件类型，请以.txt/.xlsx/.csv类型的文件作为输入。')
            return None
        mol_number = len(smiles_iterator)
        print('reading completed，A total of {} molecules detected, start calculating properties...'.format(mol_number))
        print('start calculating...')
        properties_dict_list = []
        error_smi = []
        for i in tqdm(smiles_iterator):
            try:
                properties_dict_list.append(self.calculate_a_mol(i, parameter_type=parameter_type))
            except:
                error_smi.append(i)
        print('calculation completed!')
        print('start to export result to {} ...'.format(properties_file_path))
        result = pd.DataFrame(properties_dict_list)
        result.to_csv(properties_file_path, index_label='index')
        with open('error.txt', 'w') as f:
            for i in error_smi:
                f.write(i + '\n')
        print('Done!')
        return result

    def calculate_mols_mpi(self, smiles_file_path, properties_file_path='gp_3x_result_mpi.csv', parameter_type='simultaneous', n_jobs=1, batch_size='auto'):
        print('reading input file...')
        if smiles_file_path.endswith('.txt'):
            smiles_iterator = list(open(smiles_file_path))
        elif smiles_file_path.endswith('.xlsx'):
            smiles_iterator = pd.read_excel(smiles_file_path)['smiles']
        elif smiles_file_path.endswith('.csv'):
            smiles_iterator = pd.read_csv(smiles_file_path)['smiles']
        else:
            print('无法识别的文件类型，请以.txt/.xlsx/.csv类型的文件作为输入。')
            return None
        mol_number = len(smiles_iterator)
        print('reading completed，A total of {} molecules detected, start calculating properties...'.format(mol_number))
        print('start calculating...')
        task = [delayed(self.calculate_a_mol)(i, parameter_type=parameter_type) for i in smiles_iterator]
        properties_dict_list = Parallel(n_jobs=n_jobs, batch_size=batch_size)(task)
        print('calculation completed!')
        print('start to export result to {} ...'.format(properties_file_path))
        result = pd.DataFrame(properties_dict_list)
        result.to_csv(properties_file_path, index_label='index')
        print('Done!')
        return result


# if __name__ == '__main__':
#     import time
#     t1 = time.time()
#     c = Calculator()
#     c.calculate_mols_mpi(smiles_file_path='gp_3x_test_mol/SMILES.txt', n_jobs=4)
#     t2 = time.time()
#     print(t2 - t1)