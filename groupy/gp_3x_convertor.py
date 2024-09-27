import os
from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import pybel
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed


class Convertor:
    def __init__(self):
        pass

    def __repr__(self):
        return ('This is a object implemented some functions that can convert SMILES to 3D chemical files,'
                ' such as xyz, gro... or convert 3D chemical files to SMILES')

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

    @staticmethod
    def smi_to_xyz(smi, xyz_path=None):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f'can not read {smi}, please check your SMILES')
            return False

        mol_with_h = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_with_h, useRandomCoords=True)
        atom_number = len(mol_with_h.GetAtoms())
        try:
            AllChem.MMFFOptimizeMolecule(mol_with_h)
            opt = Chem.MolToMolBlock(mol_with_h)
        except ValueError:
            mol = pybel.readstring("smi", smi)
            mol.addh()
            if mol.make3D() is None:
                opt = mol.write("mol")
            else:
                print(f'Error! There is something wrong when converting {smi} to xyz file, please check it.')
                return False

        # Windows 下pybel有问题
        # try:
        #     mol = pybel.readstring("smi", smi)
        #     mol.addh()
        #     if mol.make3D() is None:
        #         opt = mol.write("mol")
        #     else:
        #         return False
        # except:
        #     AllChem.MMFFOptimizeMolecule(mol_with_h)
        #     opt = Chem.MolToMolBlock(mol_with_h)

        if xyz_path is None:
            xyz_path = smi + '.xyz'
        with open(xyz_path, 'w') as file:
            file.write('{}\n'.format(atom_number))
            file.write(smi + '\n')
            for index, i in enumerate(opt.splitlines()[4:]):
                if len(i.split()) >= 4:
                    if i.split()[3].isupper():
                        file.write(i.split()[3] + ' ')
                        file.write(i.split()[0] + ' ')
                        file.write(i.split()[1] + ' ')
                        file.write(i.split()[2] + '\n')
        return True

    def batch_smi_to_xyz(self, smiles_file_path, xyz_root_path):
        smiles_iterator = self.load_smiles_iterator(smiles_file_path=smiles_file_path)
        mol_number = len(smiles_iterator)
        zfill_number = len(str(mol_number)) + 3
        print('reading completed，A total of {} molecules detected, start making xyz files...'.format(mol_number))
        # make xyz_root_path
        if os.path.exists(xyz_root_path):
            print('xyz_root_path "{}" has been detected!'.format(xyz_root_path))
        else:
            print('xyz_root_path "{}" has not been detected, I will create it for you'.format(xyz_root_path))
            os.makedirs(xyz_root_path)
        # end

        succeed = []
        fail = []
        for (index, smi) in tqdm(enumerate(smiles_iterator)):
            smi = smi.strip()
            out_name = os.path.join(xyz_root_path, '{}.xyz'.format(str(index).zfill(zfill_number)))
            generate_success_flag = self.smi_to_xyz(smi=smi, xyz_path=out_name)

            if not generate_success_flag:
                fail.append(smi)
            else:
                succeed.append(smi)

        with open('xyz_fail.txt', 'w') as f:
            for i in fail:
                f.write(i + '\n')
        with open('xyz_succeed.txt', 'w') as f:
            for i in succeed:
                f.write(i + '\n')

        if len(fail) == 0:
            print('done! all .xyz files has been saved in {}'.format(xyz_root_path))
        else:
            print('Warning! The following SMILES fail to generate .xyz, please check...sorry(OTZ)')
            print(fail)
        return None

    def batch_smi_to_xyz_mpi(self, smiles_file_path, xyz_root_path, n_jobs=1, batch_size='auto'):
        smiles_iterator = self.load_smiles_iterator(smiles_file_path=smiles_file_path)
        mol_number = len(smiles_iterator)
        zfill_number = len(str(mol_number)) + 3
        print('reading completed，A total of {} molecules detected, start making xyz files...'.format(mol_number))
        # make xyz_root_path
        if os.path.exists(xyz_root_path):
            print('xyz_root_path "{}" has been detected!'.format(xyz_root_path))
        else:
            print('xyz_root_path "{}" has not been detected, I will create it for you'.format(xyz_root_path))
            os.makedirs(xyz_root_path)
        # end

        task = [delayed(self.smi_to_xyz)(smi=smi, xyz_path=os.path.join(xyz_root_path, '{}.xyz'.format(str(index).zfill(zfill_number)))) for (index, smi) in enumerate(smiles_iterator)]
        result = Parallel(n_jobs=n_jobs, batch_size=batch_size)(task)
        print('done! all .xyz files has been saved in {}'.format(xyz_root_path))
        return result

    @staticmethod
    def convert_file_type(in_format, in_path, out_format, out_path=None):
        try:
            mol = pybel.readfile(in_format, in_path).__next__()
            # print('The SMILES of this system is :')
            # print(mol.write('smi'))

            if out_path is None:
                out_path = in_path.split('.')
                out_path = out_path[0] + '.' + out_format
            mol.write(out_format, out_path, overwrite=True)
            return None
        except:
            print(f'Error! There is something wrong when converting {in_path}, please check it.')
            return None

    def batch_convert_file_type(self, in_format, in_root_path, out_format, out_root_path=None):
        if out_root_path is None:
            out_root_path = in_root_path
        else:
            # make out_root_path
            if os.path.exists(out_root_path):
                print('out_root_path "{}" has been detected!'.format(out_root_path))
            else:
                print('out_root_path "{}" has not been detected, I will create it for you'.format(out_root_path))
                os.makedirs(out_root_path)
            # end
        in_file_names = os.listdir(in_root_path)

        in_file_names = [i for i in in_file_names if i.endswith(in_format)]
        out_file_names = [i.split('.')[0] + '.{}'.format(out_format) for i in in_file_names]

        in_file_path = [os.path.join(in_root_path, i) for i in in_file_names]
        out_file_path = [os.path.join(out_root_path, i) for i in out_file_names]

        error_in_file_path = []
        for index in tqdm(range(len(in_file_path))):
            try:
                self.convert_file_type(in_format=in_format, in_path=in_file_path[index],
                                       out_format=out_format, out_path=out_file_path[index])
            except:
                # print('Warning!!!')
                error_in_file_path.append(in_file_path[index])
                # print('There may something wrong in {}, please check it carefully!'.format(in_file_path[index]))

        # When there is something wrong, print some warning
        if len(error_in_file_path) > 0:
            print('Warning!Warning!Warning!')
            for i in error_in_file_path:
                print('There may something wrong in {}, please check it carefully!'.format(i))
        return None

    def batch_convert_file_type_mpi(self, in_format, in_root_path, out_format, out_root_path=None, n_jobs=1, batch_size='auto'):
        if out_root_path is None:
            out_root_path = in_root_path
        else:
            # make out_root_path
            if os.path.exists(out_root_path):
                print('out_root_path "{}" has been detected!'.format(out_root_path))
            else:
                print('out_root_path "{}" has not been detected, I will create it for you'.format(out_root_path))
                os.makedirs(out_root_path)
            # end
        in_file_names = os.listdir(in_root_path)

        in_file_names = [i for i in in_file_names if i.endswith(in_format)]
        out_file_names = [i.split('.')[0] + '.{}'.format(out_format) for i in in_file_names]

        in_file_path = [os.path.join(in_root_path, i) for i in in_file_names]
        out_file_path = [os.path.join(out_root_path, i) for i in out_file_names]

        task = [delayed(self.convert_file_type)(in_format=in_format, in_path=in_file_path[index],out_format=out_format, out_path=out_file_path[index]) for index in range(len(in_file_path))]
        result = Parallel(n_jobs=n_jobs, batch_size=batch_size)(task)
        return result

    @staticmethod
    def file_to_smi(file_path, format=None):
        """
        {'abinit': 'ABINIT Output Format', 'acesout': 'ACES output format', 'acr': 'ACR format',
        'adfband': 'ADF Band output format', 'adfdftb': 'ADF DFTB output format', 'adfout': 'ADF output format',
        'alc': 'Alchemy format', 'aoforce': 'Turbomole AOFORCE output format',
        'arc': 'Accelrys/MSI Biosym/Insight II CAR format', 'axsf': 'XCrySDen Structure Format',
        'bgf': 'MSI BGF format', 'box': 'Dock 3.5 Box format', 'bs': 'Ball and Stick format',
        'c09out': 'Crystal 09 output format', 'c3d1': 'Chem3D Cartesian 1 format', 'c3d2': 'Chem3D Cartesian 2 format',
        'caccrt': 'Cacao Cartesian format', 'can': 'Canonical SMILES format',
        'car': 'Accelrys/MSI Biosym/Insight II CAR format', 'castep': 'CASTEP format', 'ccc': 'CCC format',
        'cdjson': 'ChemDoodle JSON', 'cdx': 'ChemDraw binary format', 'cdxml': 'ChemDraw CDXML format',
        'cif': 'Crystallographic Information File', 'ck': 'ChemKin format', 'cml': 'Chemical Markup Language',
        'cmlr': 'CML Reaction format', 'cof': 'Culgi object file format', 'CONFIG': 'DL-POLY CONFIG',
        'CONTCAR': 'VASP format', 'CONTFF': 'MDFF format', 'crk2d': 'Chemical Resource Kit diagram(2D)',
        'crk3d': 'Chemical Resource Kit 3D format', 'ct': 'ChemDraw Connection Table format',
        'cub': 'Gaussian cube format', 'cube': 'Gaussian cube format', 'dallog': 'DALTON output format',
        'dalmol': 'DALTON input format', 'dat': 'Generic Output file format', 'dmol': 'DMol3 coordinates format',
        'dx': 'OpenDX cube format for APBS', 'ent': 'Protein Data Bank format',
        'exyz': 'Extended XYZ cartesian coordinates format', 'fa': 'FASTA format', 'fasta': 'FASTA format',
        'fch': 'Gaussian formatted checkpoint file format', 'fchk': 'Gaussian formatted checkpoint file format',
        'fck': 'Gaussian formatted checkpoint file format', 'feat': 'Feature format', 'fhiaims': 'FHIaims XYZ format',
        'fract': 'Free Form Fractional format', 'fs': 'Fastsearch format', 'fsa': 'FASTA format',
        'g03': 'Gaussian Output', 'g09': 'Gaussian Output', 'g16': 'Gaussian Output',
        'g92': 'Gaussian Output', 'g94': 'Gaussian Output', 'g98': 'Gaussian Output', 'gal': 'Gaussian Output',
        'gam': 'GAMESS Output', 'gamess': 'GAMESS Output', 'gamin': 'GAMESS Input', 'gamout': 'GAMESS Output',
        'got': 'GULP format', 'gpr': 'Ghemical format', 'gro': 'GRO format', 'gukin': 'GAMESS-UK Input',
        'gukout': 'GAMESS-UK Output', 'gzmat': 'Gaussian Z-Matrix Input', 'hin': 'HyperChem HIN format',
        'HISTORY': 'DL-POLY HISTORY', 'inchi': 'InChI format', 'inp': 'GAMESS Input', 'ins': 'ShelX format',
        'jin': 'Jaguar input format', 'jout': 'Jaguar output format', 'log': 'Generic Output file format',
        'lpmd': 'LPMD format', 'mcdl': 'MCDL format', 'mcif': 'Macromolecular Crystallographic Info',
        'MDFF': 'MDFF format', 'mdl': 'MDL MOL format', 'ml2': 'Sybyl Mol2 format',
        'mmcif': 'Macromolecular Crystallographic Info', 'mmd': 'MacroModel format', 'mmod': 'MacroModel format',
        'mol': 'MDL MOL format', 'mol2': 'Sybyl Mol2 format', 'mold': 'Molden format', 'molden': 'Molden format',
        'molf': 'Molden format', 'moo': 'MOPAC Output format', 'mop': 'MOPAC Cartesian format',
        'mopcrt': 'MOPAC Cartesian format', 'mopin': 'MOPAC Internal', 'mopout': 'MOPAC Output format',
        'mpc': 'MOPAC Cartesian format', 'mpo': 'Molpro output format', 'mpqc': 'MPQC output format',
        'mrv': 'Chemical Markup Language', 'msi': 'Accelrys/MSI Cerius II MSI format', 'nwo': 'NWChem output format',
        'orca': 'ORCA output format', 'out': 'Generic Output file format', 'outmol': 'DMol3 coordinates format',
        'output': 'Generic Output file format', 'pc': 'PubChem format', 'pcjson': 'PubChem JSON',
        'pcm': 'PCModel Format', 'pdb': 'Protein Data Bank format', 'pdbqt': 'AutoDock PDBQT format',
        'png': 'PNG 2D depiction', 'pos': 'POS cartesian coordinates format', 'POSCAR': 'VASP format',
        'POSFF': 'MDFF format', 'pqr': 'PQR format', 'pqs': 'Parallel Quantum Solutions format',
        'prep': 'Amber Prep format', 'pwscf': 'PWscf format', 'qcout': 'Q-Chem output format', 'res': 'ShelX format',
        'rsmi': 'Reaction SMILES format', 'rxn': 'MDL RXN format', 'sd': 'MDL MOL format', 'sdf': 'MDL MOL format',
        'siesta': 'SIESTA format', 'smi': 'SMILES format', 'smiles': 'SMILES format',
        'smy': 'SMILES format using Smiley parser', 'sy2': 'Sybyl Mol2 format', 't41': 'ADF TAPE41 format',
        'tdd': 'Thermo format', 'text': 'Read and write raw text', 'therm': 'Thermo format',
        'tmol': 'TurboMole Coordinate format', 'txt': 'Title format', 'txyz': 'Tinker XYZ format',
        'unixyz': 'UniChem XYZ format', 'VASP': 'VASP format', 'vmol': 'ViewMol format',
        'wln': 'Wiswesser Line Notation', 'xml': 'General XML format', 'xsf': 'XCrySDen Structure Format',
        'xyz': 'XYZ cartesian coordinates format', 'yob': 'YASARA.org YOB format'}
        """
        try:
            atoms = next(pybel.readfile(format=format, filename=file_path))
            smi = atoms.write(format='smi').split('\t')[0]
            # print(smi)
            return smi
        except:
            print('There may something wrong in {}, please check it carefully!'.format(file_path))
            return 'There may something wrong in {}, please check it carefully!'.format(file_path)

    def batch_file_to_smi(self, in_format, in_root_path, out_root_path=None):
        if out_root_path is None:
            out_root_path = in_root_path
        else:
            # make out_root_path
            if os.path.exists(out_root_path):
                print('out_root_path "{}" has been detected!'.format(out_root_path))
            else:
                print('out_root_path "{}" has not been detected, I will create it for you'.format(out_root_path))
                os.makedirs(out_root_path)
            # end
        in_file_names = os.listdir(in_root_path)
        in_file_names = [i for i in in_file_names if i.endswith(in_format)]
        in_file_path = [os.path.join(in_root_path, i) for i in in_file_names]
        error_in_file_path = []
        smi_list = []
        for index in tqdm(range(len(in_file_path))):
            try:
                smi_list.append(self.file_to_smi(format=in_format, file_path=in_file_path[index]))
            except:
                # print('Warning!!!')
                error_in_file_path.append(in_file_path[index])
                # print('There may something wrong in {}, please check it carefully!'.format(in_file_path[index]))

        # When there is something wrong, print some warning
        if len(error_in_file_path) > 0:
            print('Warning!Warning!Warning!')
            for i in error_in_file_path:
                print('There may something wrong in {}, please check it carefully!'.format(i))

        with open(os.path.join(out_root_path, 'SMILES.txt'), 'w') as f:
            for i in smi_list:
                f.write(i + '\n')
        return smi_list

    def batch_file_to_smi_mpi(self, in_format, in_root_path, out_root_path=None, n_jobs=1, batch_size='auto'):
        if out_root_path is None:
            out_root_path = in_root_path
        else:
            # make out_root_path
            if os.path.exists(out_root_path):
                print('out_root_path "{}" has been detected!'.format(out_root_path))
            else:
                print('out_root_path "{}" has not been detected, I will create it for you'.format(out_root_path))
                os.makedirs(out_root_path)
            # end
        in_file_names = os.listdir(in_root_path)
        in_file_names = [i for i in in_file_names if i.endswith(in_format)]
        in_file_path = [os.path.join(in_root_path, i) for i in in_file_names]

        task = [delayed(self.file_to_smi)(format=in_format, file_path=in_file_path[index]) for index in range(len(in_file_path))]
        smi_list = Parallel(n_jobs=n_jobs, batch_size=batch_size)(task)

        with open(os.path.join(out_root_path, 'SMILES.txt'), 'w') as f:
            for i in smi_list:
                f.write(i + '\n')
        return smi_list
    

# if __name__ == '__main__':
#     import time
#
#     t1 = time.time()
#     c = Convertor()
#     # c.smi_to_xyz('C1CCCC1C', 'C1CCCC1C.xyz')
#     # c.convert_file_type(in_format='xyz', in_path='C1CCCC1C.xyz', out_format='mol', out_path='C1CCCC1C.mol')
#     # c.file_to_smi('C1CCCC1C.mol', format='mol')
#
#     # c.batch_file_to_smi(in_format='mol2', in_root_path=os.path.join('gp_3x_test_mol', 'test_mol'))
#     x = c.batch_file_to_smi_mpi(in_format='mol', in_root_path='./test', n_jobs=4, batch_size='auto')
#     t2 = time.time()
#     print(t2 - t1)