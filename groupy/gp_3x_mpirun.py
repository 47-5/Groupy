from mpi4py import MPI
import numpy as np
import time
import pandas as pd
import os
import argparse

from gp_3x_calculator import Calculator
from gp_3x_counter import Counter
from gp_3x_tool import Tool


def split_smiles_list(smiles_file_path, sections):
    print('reading input file...')
    if smiles_file_path.endswith('.txt'):
        smiles_iterator = list(open(smiles_file_path))
    elif smiles_file_path.endswith('.xlsx'):
        smiles_iterator = pd.read_excel(smiles_file_path)['smiles']
    elif smiles_file_path.endswith('.csv'):
        smiles_iterator = pd.read_csv(smiles_file_path)['smiles']
    else:
        raise NotImplemented('无法识别的文件类型，请以.txt/.xlsx/.csv类型的文件作为输入。')

    smiles_iterator = [(index, i.strip()) for index, i in enumerate(smiles_iterator)]
    mol_number = len(smiles_iterator)
    print('reading completed，A total of {} molecules detected...'.format(mol_number))
    sub_mol_number = int(mol_number / sections) + 1

    split_result = [smiles_iterator[i: i + sub_mol_number] for i in range(0, mol_number, sub_mol_number)]
    return split_result  # 如果不能整分，最后一个子列表是最少的，如78，78，78，78，74


def gp_3x_mpi_calculate(smiles_file_path, result_file_path='result_mpi.csv'):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    rank_number = comm.Get_size()  # number of rank

    # 初始化不同进程，在rank0里读取原始数据
    c = Calculator()
    if rank == 0:
        t1 = time.time()
        data = split_smiles_list(smiles_file_path=smiles_file_path, sections=rank_number)
    else:
        data = None

    # 将rank0中读取的数据分发至不同进程中去(rank1 - rankn)
    recvbuf = comm.scatter(data, root=0)
    # print('rank:{} | data:{}'.format(rank, recvbuf))
    sub_result = []
    sub_error_smi = []
    for (index, i) in recvbuf:
        try:
            sub_result.append(c.calculate_a_mol(i))
        except:
            sub_error_smi.append(i)

    # 将不同进程中(rank1 - rankn)计算的结果汇总到rank0中去
    results = comm.gather(sub_result, root=0)
    error_smi = comm.gather(sub_error_smi, root=0)
    order_results = []
    # 排序 主要不知道gather会不会自动排序
    if rank == 0:
        # print(results)
        for d in data:
            order_i = d[0][1]  # d[0][1] 是要去每个子列表的第一个元组的SMILES（因为(index, smiles)）
            for sub_res in results:
                if sub_res[0]['smiles'].strip() == order_i.strip():
                    order_results += sub_res
                    break

        with open('error.txt', 'w') as f:
            print(error_smi)
            for sub_error in error_smi:
                for j in sub_error:
                    print(j)
                    f.write(j + '\n')

        print('calculation completed!')
        print('start to export result to {} ...'.format(result_file_path))
        result = pd.DataFrame(order_results)
        result.to_csv(result_file_path, index_label='index')
        print(f'Done in {time.time() - t1} s!')
    return order_results


def gp_3x_mpi_count(smiles_file_path, result_file_path='result_mpi.csv'):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    rank_number = comm.Get_size()  # number of rank

    # 初始化不同进程，在rank0里读取原始数据
    c = Counter()
    if rank == 0:
        t1 = time.time()
        data = split_smiles_list(smiles_file_path=smiles_file_path, sections=rank_number)
    else:
        data = None

    # 将rank0中读取的数据分发至不同进程中去(rank1 - rankn)
    recvbuf = comm.scatter(data, root=0)
    # print('rank:{} | data:{}'.format(rank, recvbuf))
    sub_result = [c.count_a_mol(i, add_note=True, add_smiles=True) for (index, i) in recvbuf]

    # 将不同进程中(rank1 - rankn)计算的结果汇总到rank0中去
    results = comm.gather(sub_result, root=0)
    order_results = []
    # 排序 主要不知道gather会不会自动排序
    if rank == 0:
        for d in data:
            order_i = d[0][1]
            for sub_res in results:
                if sub_res[0]['smiles'].strip() == order_i.strip():
                    order_results += sub_res
                    break

        print('calculation completed!')
        print('start to export result to {} ...'.format(result_file_path))
        result = pd.DataFrame(order_results)
        result.to_csv(result_file_path, index_label='index')
        print(f'Done in {time.time() - t1} s!')
    return order_results


def gp_3x_mpi_generate_xyz(smiles_file_path, xyz_root_path='mpi_xyz'):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    rank_number = comm.Get_size()  # number of rank

    # 初始化不同进程，在rank0里读取原始数据
    t = Tool()
    if rank == 0:
        t1 = time.time()
        # make xyz_root_path
        if not os.path.exists(xyz_root_path):
            os.makedirs(xyz_root_path)

        data = split_smiles_list(smiles_file_path=smiles_file_path, sections=rank_number)
    else:
        data = None

    # 将rank0中读取的数据分发至不同进程中去(rank1 - rankn)
    recvbuf = comm.scatter(data, root=0)
    # print('rank:{} | data:{}'.format(rank, recvbuf))

    sub_succeed = []
    sub_fail = []
    for index, i in recvbuf:
        out_path = os.path.join(xyz_root_path, '{}.xyz'.format(str(index).zfill(10)))
        generate_success_flag = t.smi_to_xyz(smi=i, xyz_path=out_path)

        if not generate_success_flag:
            sub_fail.append(i)
        else:
            sub_succeed.append(i)

    print('rank {} done!'.format(rank))

    # 将不同进程中(rank1 - rankn)计算的结果汇总到rank0中去
    succeed_ = comm.gather(sub_succeed, root=0)
    fail_ = comm.gather(sub_fail, root=0)
    if rank == 0:
        succeed = []
        fail = []
        for s, f in zip(succeed_, fail_):
            succeed += s
            fail += f
        with open('mpi_xyz_fail.txt', 'w') as f:
            for i in fail:
                f.write(i + '\n')
        with open('mpi_xyz_succeed.txt', 'w') as f:
            for i in succeed:
                f.write(i + '\n')
        if len(fail) == 0:
            print('done! all .xyz files has been saved in {}'.format(xyz_root_path))
        else:
            print('Warning! The following SMILES fail to generate .xyz, please check...sorry(OTZ)')
            print(fail)
        print(f'Done in {time.time() - t1} s!')


def gp_3x_mpi_convert_file_type(in_format, in_root_path, out_format, out_root_path=None):
    # todo
    pass


def gp_3x_mpi_generate_gjf(smiles_file_path, gjf_root_path='mpi_gjf',
                           nproc='12', mem='12GB', gaussian_keywords=None, add_other_std_tasks=False):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    rank_number = comm.Get_size()  # number of rank

    # 初始化不同进程，在rank0里读取原始数据
    t = Tool()
    if rank == 0:
        t1 = time.time()
        # make gjf_root_path
        if not os.path.exists(gjf_root_path):
            os.makedirs(gjf_root_path)
        data = split_smiles_list(smiles_file_path=smiles_file_path, sections=rank_number)
    else:
        data = None

    # 将rank0中读取的数据分发至不同进程中去(rank1 - rankn)
    recvbuf = comm.scatter(data, root=0)
    # print('rank:{} | data:{}'.format(rank, recvbuf))

    sub_succeed = []
    sub_fail = []
    for index, i in recvbuf:
        # i = i.strip() 此工作已在split_smiles_list中实现
        chk_path = '{}.chk'.format(str(index).zfill(10))
        gjf_path = os.path.join(gjf_root_path, '{}.gjf'.format(str(index).zfill(10)))
        generate_success_flag = t.smi_to_gjf(smi=i, gjf_path=gjf_path, chk_path=chk_path, nproc=nproc, mem=mem,
                                             gaussian_keywords=gaussian_keywords,
                                             add_other_std_tasks=add_other_std_tasks)
        if not generate_success_flag:
            sub_fail.append(i)
        else:
            sub_succeed.append(i)

    print('rank {} done!'.format(rank))

    # 将不同进程中(rank1 - rankn)计算的结果汇总到rank0中去
    succeed_ = comm.gather(sub_succeed, root=0)
    fail_ = comm.gather(sub_fail, root=0)
    if rank == 0:
        succeed = []
        fail = []
        for s, f in zip(succeed_, fail_):
            succeed += s
            fail += f
        with open('mpi_gjf_fail.txt', 'w') as f:
            for i in fail:
                f.write(i + '\n')
        with open('mpi_gjf_succeed.txt', 'w') as f:
            for i in succeed:
                f.write(i + '\n')
        if len(fail) == 0:
            print('done! all .gjf files has been saved in {}'.format(gjf_root_path))
        else:
            print('Warning! The following SMILES fail to generate .gjf, please check...sorry(OTZ)')
            print(fail)
        print(f'Done in {time.time() - t1} s!')
    return None


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-smiles_file_path', dest='smiles_file_path',
                        default=os.path.join('gp_3x_internal_data', 'gdb.txt'), type=str,
                        help=r'filepath of a file in which save molecules, e.g. -smiles_file_path your\target\path')
    parser.add_argument('-result_file_path', dest='result_file_path', default='mpi_batch_results.csv', type=str,
                        help=r'path of results, e.g. -result_file_path your\target\path')
    parser.add_argument('-task', dest='task', default=None, type=str,
                        help=r'task to do, must in ["calculate", "count", "xyz", "gjf"]. e.g. -calculate')
    parser.add_argument('-out_root_path', dest='out_root_path', default='mpi_out_root', type=str,
                        help=r'root path of output .xyz/.gjf files, that is, all the output .xyz/.gjf file be saved at this.'
                             r'Only useful when task is set to xyz/gjf. e.g. -out_root_path mpi_out_root')
    parser.add_argument('-nproc', dest='nproc', default='12', type=str,
                        help=r'number of cores you want gaussian to use. Only useful when task is set to gjf.'
                             r'e.g. -nproc 12')
    parser.add_argument('-mem', dest='mem', default='12GB', type=str,
                        help=r'size of memory you want gaussian to use. Only useful when task is set to gjf.'
                             r'e.g. -mem 12GB')
    parser.add_argument('-gaussian_keywords', dest='gaussian_keywords', default='#p B3LYP/6-31g*', type=str,
                        help=r'keyword for gaussian to specify the task to do. Only useful when task is set to gjf.'
                             r'e.g. -gaussian_keywords "#p B3LYP/6-31g*" ')
    parser.add_argument('--add_other_std_tasks', action='store_true',
                        help='When want to set True, input --add_other_std_tasks,'
                             'When want to set False, do not input this option')

    # 下面5行代码用来接收直接在shell、Anaconda prompt中运行时指定的参数
    args = parser.parse_args()
    TASK = args.task
    SMILES_FILE_PATH = args.smiles_file_path
    RESULT_FILE_PATH = args.result_file_path
    OUT_ROOT_PATH = args.out_root_path
    NPROC = args.nproc
    MEM = args.mem
    GAUSSIAN_KEYWORDS = args.gaussian_keywords
    ADD_OTHER_STD_TASKS = args.add_other_std_tasks

    # SMILES_PATH = os.path.join('gp_3x_internal_data', 'gdb.txt')
    if TASK in ['calculate', 'cal']:
        gp_3x_mpi_calculate(smiles_file_path=SMILES_FILE_PATH, result_file_path=RESULT_FILE_PATH)
    elif TASK in ['count']:
        gp_3x_mpi_count(smiles_file_path=SMILES_FILE_PATH, result_file_path=RESULT_FILE_PATH)
    elif TASK in ['xyz']:
        gp_3x_mpi_generate_xyz(smiles_file_path=SMILES_FILE_PATH, xyz_root_path=OUT_ROOT_PATH)
    elif TASK in ['gjf']:
        gp_3x_mpi_generate_gjf(smiles_file_path=SMILES_FILE_PATH, gjf_root_path=OUT_ROOT_PATH,
                               nproc=NPROC, mem=MEM, gaussian_keywords=GAUSSIAN_KEYWORDS,
                               add_other_std_tasks=ADD_OTHER_STD_TASKS)

