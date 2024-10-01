from rdkit import Chem

from groupy.gp_calculator import Calculator
from groupy.gp_counter import Counter
from groupy.gp_viewer import Viewer
from groupy.gp_convertor import Convertor
from groupy.gp_generator import Generator
from groupy.gp_tool import Tool, export_a_dict, logo


def main_function_5():

    while True:
        flag_file = input(
            '\n'
            '--------------------------------------------------------------------------------- \n'
            'You are in main function 5 \n'
            'what to do? \n'
            'help. print all supported file formats on screen.'
            ' 0. return to main interface. \n'
            ' 1. generate a .xyz file by input SMILES of a molecule. \n'
            ' 2. generate a batch of .xyz files.                    -2. use mpi to accelerate.\n'
            ' 3. convert a file to other format (e.g. xyz, mol, mol2, pdb...) \n'
            ' 4. convert a batch of files to other format.          -4. use mpi to accelerate.\n'
            ' 5. convert a file to SMILES. \n'
            ' 6. convert a batch of files to SMILES                 -6. use mpi to accelerate.\n'
            ' 7. generate .gjf(input file of gaussian) file by input SMILES of a molecule. \n'
            ' 8. generate a batch of .gjf files.                    -8. use mpi to accelerate. \n'
            '--------------------------------------------------------------------------------- \n'
        )

        if flag_file == 'help':
            c = Convertor()
            c.plot_supported_format()

        elif flag_file == '0':
            break

        elif flag_file == '1':
            sub_function_1_of_main_function_5()

        elif flag_file == '2':
            sub_function_2_of_main_function_5()

        elif flag_file == '-2':
            sub_function_minus_2_of_main_function_5()

        elif flag_file == '3':
            sub_function_3_of_main_function_5()

        elif flag_file == '4':
            sub_function_4_of_main_function_5()

        elif flag_file == '-4':
            sub_function_minus_4_of_main_function_5()

        elif flag_file == '5':
            sub_function_5_of_main_function_5()

        elif flag_file == '6':
            sub_function_6_of_main_function_5()

        elif flag_file == '-6':
            sub_function_minus_6_of_main_function_5()

        elif flag_file == '7':
            sub_function_7_of_main_function_5()

        elif flag_file == '8':
            sub_function_8_of_main_function_5()

        elif flag_file == '-8':
            sub_function_minus_8_of_main_function_5()

    return None


def sub_function_1_of_main_function_5():
    """
    1. generate a .xyz file by input SMILES of a molecule.
    """
    smiles = input('input the SMILES of a molecule. \n')
    smiles = smiles.strip()
    xyz_file = input(
        'please input the path of output .xyz file. If press Enter directly, {}.xyz will be used\n'.format(smiles))
    if not xyz_file:
        xyz_file = '{}.xyz'.format(smiles)
    convertor = Convertor()
    convertor.smi_to_xyz(smi=smiles, xyz_path=xyz_file)
    print('Done!')
    print('\n\n\n')
    return None


def sub_function_2_of_main_function_5():
    """
    2. generate a batch of .xyz files.
    """
    smiles_file_path = input('input a filepath of a file in which save SMILES of molecules. '
                             'e.g. ./gp_3x_test_mol/SMILES.txt \n'
                             'Hint1: Pay attention to the difference of path format in Linux and Windows! \n'
                             'Hint2: The file must not have blank line! \n')
    xyz_root_path = input('input the root path of output xyz files, that is, '
                          'all the output xyz files will be make in this path. e.g. test_xyz \n'
                          'Hint1: Pay attention to the difference of path format in Linux and Windows! \n')
    convertor = Convertor()
    convertor.batch_smi_to_xyz(smiles_file_path=smiles_file_path, xyz_root_path=xyz_root_path)
    print('\n\n\n')
    return None


def sub_function_minus_2_of_main_function_5():
    """
    2. generate a batch of .xyz files. use mpi
    """
    smiles_file_path = input('input a filepath of a file in which save SMILES of molecules. '
                             'e.g. ./gp_3x_test_mol/SMILES.txt \n'
                             'Hint1: Pay attention to the difference of path format in Linux and Windows! \n'
                             'Hint2: The file must not have blank line! \n')
    xyz_root_path = input('input the root path of output xyz files, that is, '
                          'all the output xyz files will be make in this path. e.g. test_xyz \n'
                          'Hint1: Pay attention to the difference of path format in Linux and Windows! \n')
    n_jobs = int(input('input number of cores to use. e.g. 4 \n'))
    batch_size = input('input batch size for task decomposition. e.g. 20, you can also enter "auto" \n')
    try:
        batch_size = int(batch_size)
    except:
        pass
    convertor = Convertor()
    convertor.batch_smi_to_xyz_mpi(smiles_file_path=smiles_file_path, xyz_root_path=xyz_root_path,
                                   n_jobs=n_jobs, batch_size=batch_size)
    print('\n\n\n')
    return None


def sub_function_3_of_main_function_5():
    """
    3. convert a file to other format (e.g. xyz, mol, mol2, pdb...)
    """
    in_format = input('please input the format of your input file (e.g. xyz, pdb...) \n')
    in_path = input('please input the path of input file, e.g. C1CCC1.xyz \n')
    out_format = input('please input the format of output file you want (e.g. xyz, mol2...) \n')
    out_path = input('please input the path of output file, e.g C1CCC1.mol2. \n')
    convertor = Convertor()
    convertor.convert_file_type(in_format=in_format, in_path=in_path, out_format=out_format, out_path=out_path)
    print('Done!')
    print('\n\n\n')


def sub_function_4_of_main_function_5():
    """
    4. convert a batch of file to other format.
    """
    in_format = input('please input the format of your input file (e.g. xyz, pdb...) \n')
    in_root_path = input('please input the root path of input files, that is, '
                         'all input files you want to convert should be in there.'
                         'e.g. test_xyz \n')
    out_format = input('please input the format of output file you want (e.g. xyz, mol2...) \n')
    default_out_root_path = in_root_path
    out_root_path = input('please input the root path of output file, that is, '
                          'all the output files will be saved in there\n'
                          'If press Enter directly, {} will be used \n'.format(default_out_root_path))
    if not out_root_path:
        out_root_path = default_out_root_path
    convertor = Convertor()
    convertor.batch_convert_file_type(in_format=in_format, in_root_path=in_root_path,
                                      out_format=out_format, out_root_path=out_root_path)
    print('Done!')
    print('\n\n\n')
    return None


def sub_function_minus_4_of_main_function_5():
    """
    -4. convert a batch of file to other format. use mpi.
    """
    in_format = input('please input the format of your input file (e.g. xyz, pdb...) \n')
    in_root_path = input('please input the root path of input files, that is, '
                         'all input files you want to convert should be in there.'
                         'e.g. test_xyz \n')
    out_format = input('please input the format of output file you want (e.g. xyz, mol2...) \n')
    default_out_root_path = in_root_path
    out_root_path = input('please input the root path of output file, that is, '
                          'all the output files will be saved in there\n'
                          'If press Enter directly, {} will be used \n'.format(default_out_root_path))
    if not out_root_path:
        out_root_path = default_out_root_path
    n_jobs = int(input('input number of cores to use. e.g. 4 \n'))
    batch_size = input('input batch size for task decomposition. e.g. 20, you can also enter "auto" \n')
    try:
        batch_size = int(batch_size)
    except:
        pass
    convertor = Convertor()
    convertor.batch_convert_file_type_mpi(in_format=in_format, in_root_path=in_root_path,
                                          out_format=out_format, out_root_path=out_root_path,
                                          n_jobs=n_jobs, batch_size=batch_size)
    print('Done!')
    print('\n\n\n')
    return None


def sub_function_5_of_main_function_5():
    """
    5. Converting a file into SMILES.
    """
    file_path = input('input the path of the file which you want to convert to SMILES. \n')
    format = input('input format of the file you want to convert. \n')
    convertor = Convertor()
    smi = convertor.file_to_smi(file_path=file_path, format=format)
    print(smi)
    print('Done!')
    print('\n\n\n')
    return smi


def sub_function_6_of_main_function_5():
    """
    6. Converting a batch of files into SMILES.
    """
    in_format = input('input format of the file you want to convert. \n')
    in_root_path = input('please input the root path of input files, that is, '
                         'all input files you want to convert should be in there.'
                         'e.g. test_xyz \n')
    out_root_path = input('please input the root path of output file, that is, '
                          'the output file will be saved in there.\n'
                          'If press Enter directly, out_root_path will be same as in_root_path \n')
    if not out_root_path:
        out_root_path = None
    convertor = Convertor()
    smi_list = convertor.batch_file_to_smi(in_format=in_format, in_root_path=in_root_path, out_root_path=out_root_path)
    return smi_list


def sub_function_minus_6_of_main_function_5():
    """
    -6. Converting a batch of files into SMILES with MPI acceleration.
    """
    in_format = input('input format of the file you want to convert. \n')
    in_root_path = input('please input the root path of input files, that is, '
                         'all input files you want to convert should be in there.'
                         'e.g. test_xyz \n')
    out_root_path = input('please input the root path of output file, that is, '
                          'the output file will be saved in there.\n'
                          'If press Enter directly, out_root_path will be same as in_root_path \n')
    if not out_root_path:
        out_root_path = None
    n_jobs = int(input('input number of cores to use. e.g. 4 \n'))
    batch_size = input('input batch size for task decomposition. e.g. 20, you can also enter "auto" \n')
    try:
        batch_size = int(batch_size)
    except:
        pass
    convertor = Convertor()
    smi_list = convertor.batch_file_to_smi_mpi(in_format=in_format, in_root_path=in_root_path,
                                               out_root_path=out_root_path,
                                               n_jobs=n_jobs, batch_size=batch_size)
    return smi_list

def sub_function_7_of_main_function_5():
    """
    7. generate .gjf(input file of gaussian) file by input SMILES of a molecule.
    """
    smiles = input('input the SMILES of a molecule. \n')
    smiles = smiles.strip()

    nproc = input('input the CPU cores you want to use. e.g. 12 \n')
    if not nproc:
        nproc = '12'

    mem = input('input the memory you want to use. e.g. 12GB \n')
    if not mem:
        mem = '12GB'

    chk_path = input('input the path of chk file. e.g. Cc1ccccc1.chk \n'
                     'Hint1: If press Enter directly, {}.chk will be used. \n'
                     'Hint2: Attention please! the symbol such as (, ), /, \\ and # should not appear in a filepath! \n'
                     .format(smiles))
    if not chk_path:
        chk_path = '{}.chk'.format(smiles)

    gjf_path = input('input the path of gjf file. e.g. Cc1ccccc1.gjf \n'
                     'Hint1: If press Enter directly, {}.gjf will be used. \n'
                     'Hint2: Attention please! the symbol such as (, ), /, \\ and # should not appear in a filepath! \n'
                     .format(smiles))
    if not gjf_path:
        gjf_path = '{}.gjf'.format(smiles)

    gaussian_keywords = input('input the keywords of Gaussian to define task you want to run.'
                              'e.g. #p opt freq b3lyp/6-31g* \n'
                              'Hint1: if press Enter directly, "#p opt freq b3lyp/6-31g*" will be used. \n')
    if not gaussian_keywords:
        gaussian_keywords = '#p opt freq b3lyp/6-31g*'

    charge_and_multiplicity = input('Input charge and multiplicity. e.g. 0 1 \n'
                                    'Hint: If press Enter directly, Groupy will automatically calculate them')
    if not charge_and_multiplicity:
        charge_and_multiplicity = None

    add_other_tasks = input('Weather to add some other tasks in this .gjf (y/n). \n')
    if add_other_tasks in ['n', 'no', 'N']:
        add_other_tasks = False
        other_tasks = None
    else:
        other_tasks = input('Input keywords you want to add. '
                            'If there are more than one other tasks, Please separate them with commas (,) \n'
                            'Hint: if press Enter directly, "#p m062x/def2tzvp geom=check,#p m062x/def2tzvp scrf=solvent=water geom=check" will be used \n')
        if not other_tasks:
            other_tasks = None
        else:
            other_tasks = other_tasks.split(',')

    generator = Generator()
    generator.smi_to_gjf(smi=smiles, nproc=nproc, mem=mem, gaussian_keywords=gaussian_keywords,
                         charge_and_multiplicity=charge_and_multiplicity,
                         chk_path=chk_path, gjf_path=gjf_path,
                         add_other_tasks=add_other_tasks, other_tasks=other_tasks)
    print('Done!')
    print('\n\n\n')
    return None


def sub_function_8_of_main_function_5():
    """
    8.Generating some gjf files based on a file in which saved some SMILES.
    """
    smiles_file_path = input('input the filepath of a file in which save molecules. '
                             'e.g. ./gp_3x_test_mol/SMILES.txt \n'
                             'Hint1: Pay attention to the difference of path format in Linux and Windows! \n'
                             'Hint2: The file must not have blank line! \n')

    gjf_root_path = input('Input the root path of output gjf files, that is, '
                          'all the output gjf files will be make in this path. e.g. test_gjf \n'
                          'Hint1: Pay attention to the difference of path format in Linux and Windows! \n'
                          'Hint2: if press Enter directly, test_gjf will be used. \n')
    if not gjf_root_path:
        gjf_root_path = 'test_gjf'

    nproc = input('input the CPU cores you want to use. e.g. 12 \n')
    if not nproc:
        nproc = '12'

    mem = input('input the memory you want to use. e.g. 12GB \n')
    if not mem:
        mem = '12GB'

    gaussian_keywords = input('input the keywords of Gaussian to define task you want to run.'
                              'e.g. #p opt freq b3lyp/6-31g* \n'
                              'Hint1: if press Enter directly, "#p opt freq b3lyp/6-31g*" will be used. \n')
    if not gaussian_keywords:
        gaussian_keywords = '#p opt freq b3lyp/6-31g*'

    charge_and_multiplicity = input('Input charge and multiplicity. e.g. 0 1 \n'
                                    'Hint: If press Enter directly, Groupy will automatically calculate them')
    if not charge_and_multiplicity:
        charge_and_multiplicity = None

    add_other_tasks = input('Weather to add some other tasks in this .gjf (y/n). \n')
    if add_other_tasks in ['n', 'no', 'N']:
        add_other_tasks = False
        other_tasks = None
    else:
        other_tasks = input('Input keywords you want to add. '
                            'If there are more than one other tasks, Please separate them with commas (,) \n'
                            'Hint: if press Enter directly, "#p m062x/def2tzvp geom=check,#p m062x/def2tzvp scrf=solvent=water geom=check" will be used \n')
        if not other_tasks:
            other_tasks = None
        else:
            other_tasks = other_tasks.split(',')

    generator = Generator()
    generator.batch_smi_to_gjf(smiles_file_path=smiles_file_path, gjf_root_path=gjf_root_path,
                               nproc=nproc, mem=mem, gaussian_keywords=gaussian_keywords,
                               charge_and_multiplicity=charge_and_multiplicity,
                               add_other_tasks=add_other_tasks, other_tasks=other_tasks)
    print('\n\n\n')
    return None


def sub_function_minus_8_of_main_function_5():
    """
    -8. Generating some gjf files based on a file in which saved some SMILES with MPI acceleration.
    """
    smiles_file_path = input('input the filepath of a file in which save molecules. '
                             'e.g. ./gp_3x_test_mol/SMILES.txt \n'
                             'Hint1: Pay attention to the difference of path format in Linux and Windows! \n'
                             'Hint2: The file must not have blank line! \n')

    gjf_root_path = input('Input the root path of output gjf files, that is, '
                          'all the output gjf files will be make in this path. e.g. test_gjf \n'
                          'Hint1: Pay attention to the difference of path format in Linux and Windows! \n'
                          'Hint2: if press Enter directly, test_gjf will be used. \n')
    if not gjf_root_path:
        gjf_root_path = 'test_gjf'

    nproc = input('input the CPU cores you want to use. e.g. 12 \n')
    if not nproc:
        nproc = '12'

    mem = input('input the memory you want to use. e.g. 12GB \n')
    if not mem:
        mem = '12GB'

    gaussian_keywords = input('input the keywords of Gaussian to define task you want to run.'
                              'e.g. #p opt freq b3lyp/6-31g* \n'
                              'Hint1: if press Enter directly, "#p opt freq b3lyp/6-31g*" will be used. \n')
    if not gaussian_keywords:
        gaussian_keywords = '#p opt freq b3lyp/6-31g*'

    charge_and_multiplicity = input('Input charge and multiplicity. e.g. 0 1 \n'
                                    'Hint: If press Enter directly, Groupy will automatically calculate them')
    if not charge_and_multiplicity:
        charge_and_multiplicity = None

    add_other_tasks = input('Weather to add some other tasks in this .gjf (y/n). \n')
    if add_other_tasks in ['n', 'no', 'N']:
        add_other_tasks = False
        other_tasks = None
    else:
        other_tasks = input('Input keywords you want to add. '
                            'If there are more than one other tasks, Please separate them with commas (,) \n'
                            'Hint: if press Enter directly, "#p m062x/def2tzvp geom=check,#p m062x/def2tzvp scrf=solvent=water geom=check" will be used \n')
        if not other_tasks:
            other_tasks = None
        else:
            other_tasks = other_tasks.split(',')

    n_jobs = int(input('input number of cores to use. e.g. 4 \n'))
    batch_size = input('input batch size for task decomposition. e.g. 20, you can also enter "auto" \n')
    try:
        batch_size = int(batch_size)
    except:
        pass

    generator = Generator()
    generator.batch_smi_to_gjf_mpi(smiles_file_path=smiles_file_path, gjf_root_path=gjf_root_path,
                                   nproc=nproc, mem=mem, gaussian_keywords=gaussian_keywords,
                                   charge_and_multiplicity=charge_and_multiplicity,
                                   add_other_tasks=add_other_tasks, other_tasks=other_tasks,
                                   n_jobs=n_jobs, batch_size=batch_size)
    print('\n\n\n')
    return None


def main_function_view():
    viewer = Viewer()
    show_flag = input('show a SMILES (enter 1) or file (enter 2). \n(enter help to show supported file formats)\n')
    if show_flag in ['1', 'SMILES', 'smiles']:
        smiles = input('input the SMILES of a molecule. \n')
        viewer.view_mol(mol=smiles, mol_type='smi')

    elif show_flag in ['2', 'file', 'FILE']:
        file_path = input('input the file path you want to show. e.g. ./temporary.xyz \n')
        file_type = input('input file format. e.g. xyz \n')
        viewer.view_mol(mol=file_path, mol_type=file_type)

    elif show_flag in ['h', 'help', 'H', 'Help', 'HELP']:
        viewer.plot_supported_format()

    else:
        print('Unrecognized command!')
    return None


def main_function_1():
    """
    1. calculate properties of a molecule by input SMILES of this molecule.
    """
    smiles = input('input the SMILES of a molecule. \n')
    smiles = smiles.strip()
    calculator = Calculator()
    result = calculator.calculate_a_mol(smiles, debug=False)
    print(result)
    export_flag = input('Do you want to export results to a csv file? (y/n) \n')
    if export_flag in ['y', 'Y', '1']:
        export_a_dict(result_dict=result, export_path='{}_calculate.csv'.format(smiles))
        print('the results have been export to {}_calculate.csv! \n\n'.format(smiles))
    else:
        print('\n\n\n')
    return None


def main_function_2():
    """
    2. count group number of a molecule by input SMILES of this molecule.
    """
    smiles = input('input the SMILES of a molecule. \n')
    smiles = smiles.strip()
    mol = Chem.MolFromSmiles(smiles)
    counter = Counter()
    clear_mode_flag = input('clear mode? (y/n) \n')
    if clear_mode_flag in ['y', 'Y', '1']:
        result = counter.count_a_mol(mol, clear_mode=True)
    else:
        result = counter.count_a_mol(mol, clear_mode=False)
    print(result)
    export_flag = input('Do you want to export results to a file? (y/n) \n')
    if export_flag in ['y', 'Y', '1']:
        export_a_dict(result_dict=result, export_path='{}_count.csv'.format(smiles))
        print('the results have been export to {}_count.csv! \n\n'.format(smiles))
    else:
        print('\n\n\n')
    return None


def main_function_3():
    """
    3. calculate properties of a batch of molecules by input filepath of a file in which save molecules (.txt, .csv, .xlsx).
    """
    smiles_file_path = input('input the filepath of a file (.txt, .csv, .xlsx) in which save molecules. '
                             'e.g. ./gp_3x_test_mol/SMILES.txt \n'
                             'Hint1: Pay attention to the difference of path format in Linux and Windows! \n'
                             'Hint2: The file must not have blank line! \n')
    calculator = Calculator()
    calculator.calculate_mols(smiles_file_path=smiles_file_path,
                              properties_file_path='batch_calculate_results.csv')
    print('\n\n\n')
    return None


def main_function_minus_3():
    """
    -3. calculate properties of a batch of molecules by input filepath of a file in which save molecules (.txt, .csv, .xlsx).
    use mpi
    """
    smiles_file_path = input('input the filepath of a file (.txt, .csv, .xlsx) in which save molecules. '
                             'e.g. ./gp_3x_test_mol/SMILES.txt \n'
                             'Hint1: Pay attention to the difference of path format in Linux and Windows! \n'
                             'Hint2: The file must not have blank line! \n')
    n_jobs = int(input('input number of cores to use. e.g. 4 \n'))
    batch_size = input('input batch size for task decomposition. e.g. 20, you can also enter "auto" \n')
    try:
        batch_size = int(batch_size)
    except:
        pass
    calculator = Calculator()
    calculator.calculate_mols_mpi(smiles_file_path=smiles_file_path,
                                  properties_file_path='batch_calculate_results_mpi.csv',
                                  n_jobs=n_jobs, batch_size=batch_size)
    print('\n\n\n')
    return None


def main_function_4():
    """
    4. count group number of a batch of molecules by input filepath of a file in which save molecules (.txt, .csv, .xlsx).
    """
    smiles_file_path = input('input the filepath of a file in which save molecules. '
                             'e.g. ./gp_3x_test_mol/SMILES.txt \n'
                             'Hint1: Pay attention to the difference of path format in Linux and Windows! \n'
                             'Hint2: The file must not have blank line! \n')
    counter = Counter()
    counter.count_mols(smiles_file_path=smiles_file_path,
                       count_result_file_path='batch_count_result.csv', add_note=True, add_smiles=True)
    print('\n\n\n')
    return None


def main_function_minus_4():
    """
    -4. count group number of a batch of molecules by input filepath of a file in which save molecules (.txt, .csv, .xlsx).
    use mpi
    """
    smiles_file_path = input('input the filepath of a file (.txt, .csv, .xlsx) in which save molecules. '
                             'e.g. ./gp_3x_test_mol/SMILES.txt \n'
                             'Hint1: Pay attention to the difference of path format in Linux and Windows! \n'
                             'Hint2: The file must not have blank line! \n')
    n_jobs = int(input('input number of cores to use. e.g. 4 \n'))
    batch_size = input('input batch size for task decomposition. e.g. 20, you can also enter "auto" \n')
    try:
        batch_size = int(batch_size)
    except:
        pass
    counter = Counter()
    counter.count_mols_mpi(smiles_file_path=smiles_file_path,
                           count_result_file_path='batch_count_result_mpi.csv', add_note=True, add_smiles=True,
                           n_jobs=n_jobs, batch_size=batch_size)
    print('\n\n\n')
    return None


def main():
    print(logo)

    while True:

        flag_main = input(
              '\n'
              '--------------------------------------------------------------------------------- \n'
              'You are in main interface \n'
              'what to do? \n'
              ' q. exit \n'
              ' 0. show molecular structure by SMILES or file. \n'
              ' 1. calculate properties of a molecule. \n'
              ' 2. count group number of a molecule. \n'
              ' 3. calculate properties of a batch of molecules.    -3. use mpi to accelerate.\n'
              ' 4. count group number of a batch of molecules.      -4. use mpi to accelerate.\n'
              ' 5. generate files or covert file format for MD, DFT, Visualization... \n'
              '--------------------------------------------------------------------------------- \n'
        )

        if flag_main == 'q':
            print('exit Groupy, have a nice day!')
            break

        elif flag_main == '0':
            main_function_view()

        elif flag_main == '1':
            main_function_1()

        elif flag_main == '2':
            main_function_2()

        elif flag_main == '3':
            main_function_3()

        elif flag_main == '-3':
            main_function_minus_3()

        elif flag_main == '4':
            main_function_4()

        elif flag_main == '-4':
            main_function_minus_4()

        elif flag_main == '5':
            main_function_5()

        else:
            print('Please input the right option. '
                  'For more information, you are supposed to read the manual.')


if __name__ == '__main__':

    main()
