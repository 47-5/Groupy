from rdkit import Chem
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import time
import random

from groupy.gp_convertor import Convertor
from groupy.gp_tool import Tool


class Generator:
    """
    A Class for generating Gaussian Job File (gjf)
    """
    def __init__(self):
        pass

    def calculate_charge(self, smi):
        """
        Calculating net charge (all proton - all charge) and all charge.
        This method is used to calculate net charge which is used to generate gjf file.
        Generator.calculate_multiplicity() also call this method to calculate multiplicity.
        In general, users do not need to use this method.
        :param smi: SMILES str or instance of rdkit.Chem.rdchem.Mol
        :return: (int, int). (net charge, all charge)
        """
        if isinstance(smi, str):
            smi = Chem.MolFromSmiles(smi)
        net_charge = 0
        all_charge = 0
        for i in smi.GetAtoms():
            net_charge += i.GetFormalCharge()
            all_charge += i.GetAtomicNum()

        all_charge -= net_charge

        return net_charge, all_charge

    def calculate_multiplicity(self, smi):
        """
        Calculating multiplicity to generate gjf files.
        Generator.calculate_multiplicity() also call this method to calculate multiplicity.
        In general, users do not need to use this method. We also do not recommend users to explicitly use this
        function for calculating the multiplicity of their choice, as this method only considers the simplest
        scenario, where alpha electrons pair with beta electrons as much as possible.
        :param smi: SMILES str or instance of rdkit.Chem.rdchem.Mol
        :return: (int, int). (net charge, all charge)
        """
        if isinstance(smi, str):
            smi = Chem.MolFromSmiles(smi)
        net_charge, all_charge = self.calculate_charge(smi)
        alpha_minus_beta = all_charge % 2
        multiplicity = alpha_minus_beta + 1
        return multiplicity

    def smi_to_gjf(self, smi, nproc='12', mem='12GB', chk_path=None, gjf_path=None,
                   gaussian_keywords=None, charge_and_multiplicity=None,
                   add_other_tasks=False, other_tasks:list=None):
        """
        Generating a gjf file based on SMILES of molecule.
        :param smi: str. SMILES of a molecule.
        :param nproc: str or int. The number of CPU cores allowed for Gaussian to utilize. Default=12.
        :param mem: str. The amount of memory allocated for Gaussian to utilize. Default=12GB.
        :param chk_path: str. Path of chk file. Default={smi}.chk
        :param gjf_path: str. Path of gjf file you want to generate. Default={smi}.chk
        :param gaussian_keywords: str. Keywords in gjf fie. Default='#p opt freq b3lyp/6-31g*'
        :param charge_and_multiplicity: str. charge and multiplicity in gjf file. If set to None,
                                        Generator will call Generator.calculate_charge() and
                                        Generator.calculate_multiplicity() to calculate. example: '0 1'
        :param add_other_tasks: bool. Whether to add other job into you gjf file. Default=False
        :param other_tasks: list. Jobs you want to add into you gjf file. Default = ['#p m062x/def2tzvp geom=check',
                    '#p m062x/def2tzvp scrf=solvent=water geom=check',]. Note that this parameter will only be used if add_other_tasks=True.
        :return: bool. True if the gjf file is successfully generated.
        """
        try:
            # default path of chk and gjf
            if chk_path is None:
                chk_path = '{}.chk'.format(smi)
            if gjf_path is None:
                gjf_path = '{}.gjf'.format(smi)
            assert ('(' not in gjf_path) and (')' not in gjf_path), \
                'gaussian dose not allow ( or ) in the name of .gjf and .chk files'
            assert ('(' not in chk_path) and (')' not in chk_path), \
                'gaussian dose not allow ( or ) in the name of .gjf and .chk files'
            # default task
            if gaussian_keywords is None:
                gaussian_keywords = '#p opt freq b3lyp/6-31g*'
            # default charge and multiplicity
            if charge_and_multiplicity is None:
                charge_and_multiplicity = f'{self.calculate_charge(smi)[0]} {self.calculate_multiplicity(smi)}'
            # default other tasks
            if other_tasks is None:
                other_tasks = [
                    '#p m062x/def2tzvp geom=check',
                    '#p m062x/def2tzvp scrf=solvent=water geom=check',
                ]

            # read smi
            c = Convertor()
            temp_xyz_path = f'temp_{str(time.time()) + str(random.randint(0,1000000000000000000))}.xyz'
            c.smi_to_xyz(smi=smi, xyz_path=temp_xyz_path)

            # write gjf
            # 判断是否存在同名gjf
            if os.path.exists(gjf_path):
                os.remove(gjf_path)
            self.write_gjf_link0_and_keyword(gjf_path=gjf_path, chk_path=chk_path, nproc=nproc, mem=mem,
                                             gaussian_keywords=gaussian_keywords,
                                             charge_and_multiplicity=charge_and_multiplicity, note=smi)
            self.write_gjf_coord(gjf_path=gjf_path, xyz_path=temp_xyz_path)

            if add_other_tasks:
                for task_index, task in enumerate(other_tasks):
                    i_chk_path = chk_path.split('.')[0] + f'_{task_index + 1}' + '.chk'
                    self.write_gjf_link0_and_keyword(gjf_path=gjf_path, chk_path=i_chk_path, nproc=nproc, mem=mem,
                                                     gaussian_keywords=task, charge_and_multiplicity=charge_and_multiplicity,
                                                     note=smi, old_chk_path=chk_path, add_link1=True)
                    self.write_gjf_blank_line(gjf_path=gjf_path, blank_line_number=2)

            # 删除临时xyz文件
            os.remove(temp_xyz_path)
            return True
        except:
            print(f'Error! There is something wrong when converting {smi} to gjf file, please check it.')
            return False

    def batch_smi_to_gjf(self, smiles_file_path, gjf_root_path=None,
                         nproc='12', mem='12GB',
                         gaussian_keywords=None, charge_and_multiplicity=None,
                         add_other_tasks=False, other_tasks: list = None,
                         index_start=0,
                         ):
        """
        Generating some gjf files based on a file in which saved some SMILES.
        :param smiles_file_path: str. Path of the file in which saved SMILES.
        :param gjf_root_path: str. The folder path where all generated gjf files are saved.
        :param nproc: str or int. The number of CPU cores allowed for Gaussian to utilize. Default=12.
        :param mem: str. The amount of memory allocated for Gaussian to utilize. Default=12GB.
        :param gaussian_keywords: str. Keywords in gjf fie. Default='#p opt freq b3lyp/6-31g*'
        :param charge_and_multiplicity: str. charge and multiplicity in gjf file. If set to None,
                                        Generator will call Generator.calculate_charge() and
                                        Generator.calculate_multiplicity() to calculate. example: '0 1'
        :param add_other_tasks: bool. Whether to add other job into you gjf file. Default=False
        :param other_tasks: list. Jobs you want to add into you gjf file. Default = ['#p m062x/def2tzvp geom=check',
                    '#p m062x/def2tzvp scrf=solvent=water geom=check',]. Note that this parameter will only be used if add_other_tasks=True.
        :return: None
        """
        smiles_iterator = Tool.load_smiles_iterator(smiles_file_path=smiles_file_path)
        mol_number = len(smiles_iterator)
        zfill_number = len(str(mol_number)) + 5
        print('reading completed，A total of {} molecules detected, start calculating properties...'.format(mol_number))

        # make gjf_root_path
        if os.path.exists(gjf_root_path):
            print('gjf_root_path "{}" has been detected!'.format(gjf_root_path))
        else:
            print('gjf_root_path "{}" has not been detected, I will create it for you'.format(gjf_root_path))
            os.makedirs(gjf_root_path)
        # end

        succeed = []
        fail = []
        for (index, smi) in tqdm(enumerate(smiles_iterator)):
            smi = smi.strip()
            index += index_start
            chk_path = '{}.chk'.format(str(index).zfill(zfill_number))
            gjf_path = os.path.join(gjf_root_path, '{}.gjf'.format(str(index).zfill(zfill_number)))
            generate_success_flag = self.smi_to_gjf(smi=smi, nproc=nproc, mem=mem,
                                                    chk_path=chk_path, gjf_path=gjf_path,
                                                    gaussian_keywords=gaussian_keywords,
                                                    charge_and_multiplicity=charge_and_multiplicity,
                                                    add_other_tasks=add_other_tasks, other_tasks=other_tasks,
                                                    )
            if not generate_success_flag:
                fail.append(smi)
            else:
                succeed.append(smi)

        with open('gjf_fail.txt', 'w') as f:
            for i in fail:
                f.write(i + '\n')
        with open('gjf_succeed.txt', 'w') as f:
            for i in succeed:
                f.write(i + '\n')
        if len(fail) == 0:
            print('done! all .gjf files has been saved in {}'.format(gjf_root_path))
        else:
            print('Warning! The following SMILES fail to generate .gjf, please check...sorry(OTZ)')
            print(fail)
        return None

    def batch_smi_to_gjf_mpi(self, smiles_file_path, gjf_root_path=None,
                             nproc='12', mem='12GB',
                             gaussian_keywords=None, charge_and_multiplicity=None,
                             add_other_tasks=False, other_tasks: list = None,
                             n_jobs=1, batch_size='auto'
                         ):
        """
        Generating some gjf files based on a file in which saved some SMILES with MPI acceleration.
        :param smiles_file_path: str. Path of the file in which saved SMILES.
        :param gjf_root_path: str. The folder path where all generated gjf files are saved.
        :param nproc: str or int. The number of CPU cores allowed for Gaussian to utilize. Default=12.
        :param mem: str. The amount of memory allocated for Gaussian to utilize. Default=12GB.
        :param gaussian_keywords: str. Keywords in gjf fie. Default='#p opt freq b3lyp/6-31g*'
        :param charge_and_multiplicity: str. charge and multiplicity in gjf file. If set to None,
                                        Generator will call Generator.calculate_charge() and
                                        Generator.calculate_multiplicity() to calculate. example: '0 1'
        :param add_other_tasks: bool. Whether to add other job into you gjf file. Default=False
        :param other_tasks: list. Jobs you want to add into you gjf file. Default = ['#p m062x/def2tzvp geom=check',
                    '#p m062x/def2tzvp scrf=solvent=water geom=check',]. Note that this parameter will only be used if add_other_tasks=True.
        :param n_jobs: int. number of CPU cores you want to use when generating gjf file.
        :param batch_size: int or str. Number of tasks per CPU core you want to use when generating gjf file. Default='auto'.
        :return: None
        """
        smiles_iterator = Tool.load_smiles_iterator(smiles_file_path=smiles_file_path)
        mol_number = len(smiles_iterator)
        zfill_number = len(str(mol_number)) + 5
        print('reading completed，A total of {} molecules detected, start calculating properties...'.format(mol_number))

        # make gjf_root_path
        if os.path.exists(gjf_root_path):
            print('gjf_root_path "{}" has been detected!'.format(gjf_root_path))
        else:
            print('gjf_root_path "{}" has not been detected, I will create it for you'.format(gjf_root_path))
            os.makedirs(gjf_root_path)

        # task
        task = [delayed(self.smi_to_gjf)(smi=smi, nproc=nproc, mem=mem,
                                         chk_path='{}.chk'.format(str(index).zfill(zfill_number)),
                                         gjf_path=os.path.join(gjf_root_path, '{}.gjf'.format(str(index).zfill(zfill_number))),
                                         gaussian_keywords=gaussian_keywords,
                                         charge_and_multiplicity=charge_and_multiplicity,
                                         add_other_tasks=add_other_tasks, other_tasks=other_tasks,)
                for index, smi in enumerate(smiles_iterator)]
        result = Parallel(n_jobs=n_jobs, batch_size=batch_size)(task)
        return result

    @staticmethod
    def write_gjf_link0_and_keyword(gjf_path, chk_path, nproc, mem, gaussian_keywords, charge_and_multiplicity, note,
                                    old_chk_path=None, add_link1=False):
        """
        Only used in Generator.smi_to_gjf()
        """
        with open(gjf_path, 'a') as gjf:
            if add_link1:
                gjf.write('--link1--' + '\n')
            gjf.write(f'%nproc={nproc}' + '\n')
            gjf.write(f'%mem={mem}' + '\n')
            if old_chk_path is not None:
                gjf.write(f'%oldchk={old_chk_path}' + '\n')
            gjf.write(f'%chk={chk_path}' + '\n')
            gjf.write(f'{gaussian_keywords}' + '\n')
            gjf.write('\n')
            gjf.write(f'{note}' + '\n')
            gjf.write('\n')
            gjf.write(f'{charge_and_multiplicity}' + '\n')
            gjf.close()
        return None

    @staticmethod
    def write_gjf_coord(gjf_path, xyz_path):
        """
        Only used in Generator.smi_to_gjf()
        """
        xyz = open(xyz_path)
        with open(gjf_path, 'a') as gjf:
            for i in xyz.readlines()[2:]:
                gjf.write(i)
            gjf.write('\n\n')
        return None

    @staticmethod
    def write_gjf_blank_line(gjf_path, blank_line_number=1):
        """
        Only used in Generator.smi_to_gjf()
        """
        with open(gjf_path, 'a') as gjf:
            gjf.write('\n' * blank_line_number)
        return None





# if __name__ == '__main__':
#
#     import time
#     g = Generator()
#     t1 = time.time()
#     # g.smi_to_gjf(smi='C1CCCC1', add_other_tasks=True)
#     # g.batch_smi_to_gjf(smiles_file_path='gp_3x_test_mol/3018_with_error_smiles.txt', gjf_root_path='./test_gjf')
#     g.batch_smi_to_gjf_mpi(smiles_file_path='gp_3x_test_mol/3018_with_error_smiles.txt', gjf_root_path='./test_gjf',
#                            add_other_tasks=True,
#                            n_jobs=8, batch_size='auto')
#
#     t2 = time.time()
#     print(t2 - t1)