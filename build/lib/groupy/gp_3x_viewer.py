import ase
from ase.visualize import view
from ase.io import read
import os
from pprint import pprint

from gp_3x_convertor import Convertor


class Viewer:
    def __init__(self):
        pass

    def view_mol(self, mol, mol_type='smi'):
        if isinstance(mol, str):
            if mol_type in ['smi', 'smiles', 'SMILES']:
                convertor = Convertor()
                convertor.smi_to_xyz(smi=mol, xyz_path='temp.xyz')
                mol = read(filename='temp.xyz', format='xyz')
                os.remove('temp.xyz')
            else:
                try:
                    mol = read(filename=mol, format=mol_type)
                except:
                    convertor = Convertor()
                    convertor.convert_file_type(in_format=mol_type, in_path=mol, out_format='xyz', out_path='temp.xyz')
                    mol = read(filename='temp.xyz', format='xyz')
                    os.remove('temp.xyz')
        view(mol)

    def plot_supported_format(self):
        ase_format = ase.io.formats.ioformats
        openbabel_format = {'abinit': 'ABINIT Output Format', 'acesout': 'ACES output format', 'acr': 'ACR format',
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
        print('-' * 40)
        pprint('ASE format:')
        pprint(ase_format)
        print('-' * 40)
        pprint('OpenBabel format:')
        pprint(openbabel_format)
        return None



# if __name__ == '__main__':
#
#     viewer = Viewer()
#     viewer.view_mol('gp_3x_test_mol/test_xyz/000000.mol2', mol_type='mol2')
#     viewer.plot_supported_format()