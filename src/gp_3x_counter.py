import os.path
from tqdm import tqdm
from rdkit import Chem
import pandas as pd
from joblib import Parallel, delayed


# tool
def has_non_aromatic_neighbor(atom):
    """判断原子周围是否有非芳香原子"""
    flag = False
    neighbors = atom.GetNeighbors()
    for i in neighbors:
        if not i.GetIsAromatic():
            flag = True
            break
    return flag


def find_ring_atoms(mol, atom_idxs):
    """找出给定的原子索引元组中在环上的原子索引，并返回一个元组"""
    return list((i for i in atom_idxs if mol.GetAtomWithIdx(i).IsInRing()))


def is_in_same_ring(mol, atom_idxs: tuple):
    """判断一个元组中的原子索引是否在同一个环上"""
    flag = False
    all_rings = [list(i) for i in Chem.GetSymmSSSR(mol)]
    # set(i).issubset(j)
    for i in all_rings:
        if set(atom_idxs).issubset(i):
            flag = True
            break
    return flag


# f order
def f_001(mol):
    """CH3"""
    query = Chem.MolFromSmarts('[C;H3;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_002(mol):
    """CH2"""
    query = Chem.MolFromSmarts('[C;H2;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_003(mol):
    """CH"""
    query = Chem.MolFromSmarts('[C;H1;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_004(mol):
    """C"""
    query = Chem.MolFromSmarts('[C;H0;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_005(mol):
    """CH2=CH"""
    query = Chem.MolFromSmarts('[C;H2;!R]=[C;H1;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_006(mol):
    """CH=CH"""
    query = Chem.MolFromSmarts('[C;H;!R]=[C;H1;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_007(mol):
    """CH2=C"""
    query = Chem.MolFromSmarts('[C;H2;!R]=[C;H0;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_008(mol):
    """CH=C"""
    query = Chem.MolFromSmarts('[C;H1;!R]=[C;H0;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_009(mol):
    """C=C"""
    query = Chem.MolFromSmarts('[C;H0;!R]=[C;H0;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_010(mol):
    """CH2=C=CH"""
    query = Chem.MolFromSmarts('[C;H2;!R]=[C;H0;!R;D2]=[C;H1;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_011(mol):
    """CH2=C=C"""
    query = Chem.MolFromSmarts('[C;H2;!R]=[C;H0;!R;D2]=[C;H0;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_012(mol):
    """C=C=C
    2001年的论文12号基团是CH=C=CH,对应的SMART为[C;H1;!R;D2]=[C;H0;!R;D2]=[C;H1;!R;D2]
    """
    query = Chem.MolFromSmarts('[C;H0;!R]=[C;H0;!R;D2]=[C;H0;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_013(mol):
    """CH#C"""
    query = Chem.MolFromSmarts('[C;H1;!R;D1]#[C;H0;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_014(mol):
    """C#C"""
    query = Chem.MolFromSmarts('[C;H0;!R;D2]#[C;H0;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_015(mol):
    """aCH"""
    query = Chem.MolFromSmarts('[c;H1;R1;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_016(mol):  # 麻烦且慢，但是应该可靠
    """aC fused with aromatic ring"""
    query = Chem.MolFromSmarts('[c;H0;R&!R1;D3]')
    match_list = mol.GetSubstructMatches(query)
    real_match_list = []
    for i in match_list:
        atom_i = mol.GetAtomWithIdx(i[0])
        if not has_non_aromatic_neighbor(atom_i):  # 判断该原子是否有链接非芳香原子是为了防止把芳环上连接非芳香环也算进去
            real_match_list.append(i)
    return len(real_match_list), tuple(real_match_list)


def f_017(mol):  # 麻烦且慢，但是应该可靠
    """aC fused with nonaromatic subring"""
    query = Chem.MolFromSmarts('[c;H0;R&!R1;D3]')
    match_list = mol.GetSubstructMatches(query)
    real_match_list = []
    for i in match_list:
        atom_i = mol.GetAtomWithIdx(i[0])
        if has_non_aromatic_neighbor(atom_i):  # 判断该原子是否有链接非芳香原子是为了取出芳环上连接非芳香环的，和上一个函数刚好相反
            real_match_list.append(i)
    return len(real_match_list), tuple(real_match_list)


def f_018(mol):
    """aC except as above
    这里这样写是因为我们给出了数基团的顺序，在那里会剔除重复的，所以不用担心重复
    """
    query = Chem.MolFromSmarts('[c;H0;R;D3]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_019(mol):
    """aN in aromatic ring"""
    query = Chem.MolFromSmarts('[n;H0;R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_020(mol):
    """aC-CH3"""
    query = Chem.MolFromSmarts('[c;H0;R;D3][C;H3;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_021(mol):
    """aC-CH2"""
    query = Chem.MolFromSmarts('[c;H0;R;D3][C;H2;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_022(mol):
    """aC-CH"""
    query = Chem.MolFromSmarts('[c;H0;R;D3][C;H1;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_023(mol):
    """aC-C"""
    query = Chem.MolFromSmarts('[c;H0;R;D3][C;H0;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_024(mol):
    """aC-CH=CH2"""
    query = Chem.MolFromSmarts('[c;H0;R;D3][C;H1;!R;D2]=[C;H2;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_025(mol):
    """aC-CH=CH"""
    query = Chem.MolFromSmarts('[c;H0;R;D3][C;H1;!R;D2]=[C;H1;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_026(mol):
    """aC-C=CH2"""
    query = Chem.MolFromSmarts('[c;H0;R;D3][C;H0;!R;D3]=[C;H2;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_027(mol):
    """aC-C#CH"""
    query = Chem.MolFromSmarts('[c;H0;R;D3][C;H0;!R;D2]#[C;H1;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_028(mol):
    """aC-C#C"""
    query = Chem.MolFromSmarts('[c;H0;R;D3][C;H0;!R;D2]#[C;H0;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_029(mol):
    """OH"""
    query = Chem.MolFromSmarts('[O;H1;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_030(mol):
    """aC-OH"""
    query = Chem.MolFromSmarts('[c;H0;R;D3][O;H1;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_031(mol):
    """COOH"""
    query = Chem.MolFromSmarts('[C;H0;!R;D3]([O;H1;!R;D1])=[O;H0;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_032(mol):
    """aC-COOH"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[C;H0;!R;D3]([O;H1;!R;D1])=[O;H0;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_033(mol):
    """CH3CO"""
    query = Chem.MolFromSmarts('[C;H3;!R;D1][C;H0;!R;D3](=[O;H0;!R;D1])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_034(mol):
    """CH2CO"""
    query = Chem.MolFromSmarts('[C;H2;!R;D2][C;H0;!R;D3](=[O;H0;!R;D1])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_035(mol):
    """CHCO"""
    query = Chem.MolFromSmarts('[C;H1;!R][C;H0;!R;D3](=[O;H0;!R;D1])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_036(mol):
    """CCO"""
    query = Chem.MolFromSmarts('[C;H0;!R][C;H0;!R;D3](=[O;H0;!R;D1])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_037(mol):
    """aC-CO"""
    query = Chem.MolFromSmarts('[c;H0;R;D3][C;H0;!R;D3](=[O;H0;!R;D1])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_038(mol):
    """CHO"""
    query = Chem.MolFromSmarts('[C;H1;!R;D2](=[O;H0;!R;D1])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_039(mol):
    """aC-CHO"""
    query = Chem.MolFromSmarts('[c;H0;R;D3][C;H1;!R;D2](=[O;H0;!R;D1])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_040(mol):
    """CH3COO"""
    query = Chem.MolFromSmarts('[C;H3;!R;D1][C;H0;!R;D3](=[O;H0;!R;D1])[O;H0;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_041(mol):
    """CH2COO"""
    query = Chem.MolFromSmarts('[C;H2;!R;D2][C;H0;!R;D3](=[O;H0;!R;D1])[O;H0;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_042(mol):
    """CHCOO"""
    query = Chem.MolFromSmarts('[C;H1;!R;D3][C;H0;!R;D3](=[O;H0;!R;D1])[O;H0;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_043(mol):
    """CCOO"""
    query = Chem.MolFromSmarts('[C;H0;!R;D4][C;H0;!R;D3](=[O;H0;!R;D1])[O;H0;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_044(mol):
    """HCOO"""
    query = Chem.MolFromSmarts('[C;H1;!R;D2](=[O;H0;!R;D1])[O;H0;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_045(mol):
    """aC-COO"""
    query = Chem.MolFromSmarts('[c;H0;R;D3][C;H0;!R;D3](=[O;H0;!R;D1])[O;H0;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_046(mol):
    """aC-OOCH"""
    query = Chem.MolFromSmarts('[C;H1;!R;D2](=[O;H0;!R;D1])[O;H0;!R;D2][c;H0;R;D3]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_047(mol):
    """aC-OOC"""
    query = Chem.MolFromSmarts('[C;H0;!R;D3](=[O;H0;!R;D1])[O;H0;!R;D2][c;H0;R;D3]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_048(mol):
    """COO except as above
    取消了是不是在环上的限制
    """
    query = Chem.MolFromSmarts('[C;H0;D3](=[O;H0;D1])[O;H0;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_049(mol):
    """CH3O"""
    query = Chem.MolFromSmarts('[C;H3;!R;D1]-[O;H0;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_050(mol):
    """CH2O"""
    query = Chem.MolFromSmarts('[C;H2;!R;D2]-[O;H0;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_051(mol):
    """CH-O"""
    query = Chem.MolFromSmarts('[C;H1;!R]-[O;H0;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_052(mol):
    """C-O"""
    query = Chem.MolFromSmarts('[C;H0;!R]-[O;H0;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_053(mol):
    """aC-O"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[O;H0;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_054(mol):
    """CH2NH2"""
    query = Chem.MolFromSmarts('[C;H2;!R;D2]-[N;H2;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_055(mol):
    """CHNH2"""
    query = Chem.MolFromSmarts('[C;H1;!R;D3]-[N;H2;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_056(mol):
    """CNH2"""
    query = Chem.MolFromSmarts('[C;H0;!R;D4]-[N;H2;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_057(mol):
    """CH3NH"""
    query = Chem.MolFromSmarts('[C;H3;!R;D1]-[N;H1;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_058(mol):
    """CH2NH"""
    query = Chem.MolFromSmarts('[C;H2;!R;D2]-[N;H1;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_059(mol):
    """CHNH"""
    query = Chem.MolFromSmarts('[C;H1;!R;D3]-[N;H1;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_060(mol):
    """CH3N
    因为有f66(CH=N)、f67(C=N)，所以这里还是要求N的度为3
    """
    query = Chem.MolFromSmarts('[C;H3;!R;D1]-[N;H0;!R;D3]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_061(mol):
    """CH2N
    因为有f66(CH=N)、f67(C=N)，所以这里还是要求N的度为3
    """
    query = Chem.MolFromSmarts('[C;H2;!R;D2]-[N;H0;!R;D3]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_062(mol):
    """aC-NH2"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[N;H2;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_063(mol):
    """aC-NH"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[N;H1;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_064(mol):
    """aC-N"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[N;H0;!R;D3]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_065(mol):
    """NH2 except as above"""
    query = Chem.MolFromSmarts('[N;H2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_066(mol):
    """CH=N"""
    query = Chem.MolFromSmarts('[C;H1;!R;D2]=[N;H0;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_067(mol):
    """C=N"""
    query = Chem.MolFromSmarts('[C;H0;!R]=[N;H0;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_068(mol):
    """CH2CN"""
    query = Chem.MolFromSmarts('[C;H2;!R;D2][C;H0;!R;D2]#[N;H0;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_069(mol):
    """CHCN"""
    query = Chem.MolFromSmarts('[C;H1;!R;D3][C;H0;!R;D2]#[N;H0;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_070(mol):
    """CCN"""
    query = Chem.MolFromSmarts('[C;H0;!R;D4][C;H0;!R;D2]#[N;H0;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_071(mol):
    """aC-CN"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[C;H0;!R;D2]#[N;H0;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_072(mol):
    """CN except as above
    不可能在环上
    """
    query = Chem.MolFromSmarts('[C;H0;!R;D2]#[N;H0;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_073(mol):
    """CH2NCO"""
    query = Chem.MolFromSmarts('[C;H2;!R;D2]-[N;H0;!R;D2]=[C;H0;!R;D2]=[O;H0;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_074(mol):
    """CHNCO"""
    query = Chem.MolFromSmarts('[C;H1;!R]-[N;H0;!R;D2]=[C;H0;!R;D2]=[O;H0;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_075(mol):
    """CNCO"""
    query = Chem.MolFromSmarts('[C;H0;!R]-[N;H0;!R;D2]=[C;H0;!R;D2]=[O;H0;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_076(mol):
    """aC-NCO"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[N;H0;!R;D2]=[C;H0;!R;D2]=[O;H0;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_077(mol):
    """CH2NO2"""
    query = Chem.MolFromSmarts('[C;H2;!R;D2]-[N;+]([O;-])=[O]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_078(mol):
    """CHNO2"""
    query = Chem.MolFromSmarts('[C;H1;!R;D3]-[N;+]([O;-])=[O]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_079(mol):
    """CNO2"""
    query = Chem.MolFromSmarts('[C;H0;!R;D4]-[N;+]([O;-])=[O]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_080(mol):
    """aC-NO2"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[N;+]([O;-])=[O]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_081(mol):
    """NO2 except as above"""
    query = Chem.MolFromSmarts('[N;+]([O;-])=[O]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_082(mol):
    """ONO"""
    query = Chem.MolFromSmarts('[O;H0;!R;D2][N;H0;!R;D2]=[O;H0;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_083(mol):
    """ONO2"""
    query = Chem.MolFromSmarts('[O;H0;!R;D2][N;+]([O;-1])=O')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_084(mol):
    """HCON(CH2)2"""
    query = Chem.MolFromSmarts('[O;H0;!R;D1]=[C;H1;!R;D2][N;H0;!R;D3]([C;H2;!R;D2])[C;H2;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_085(mol):
    """HCONH(CH2)"""
    query = Chem.MolFromSmarts('[O;H0;!R;D1]=[C;H1;!R;D2][N;H1;!R;D2][C;H2;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_086(mol):
    """CONH2"""
    query = Chem.MolFromSmarts('[O;H0;!R;D1]=[C;H0;!R;D3][N;H2;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_087(mol):
    """CONHCH3"""
    query = Chem.MolFromSmarts('[O;H0;!R;D1]=[C;H0;!R;D3][N;H1;!R;D2][C;H3;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_088(mol):
    """CONHCH2"""
    query = Chem.MolFromSmarts('[O;H0;!R;D1]=[C;H0;!R;D3][N;H1;!R;D2][C;H2;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_089(mol):
    """CON(CH3)2"""
    query = Chem.MolFromSmarts('[O;H0;!R;D1]=[C;H0;!R;D3][N;H0;!R;D3]([C;H3;!R;D1])[C;H3;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_090(mol):
    """CONCH3CH2"""
    query = Chem.MolFromSmarts('[O;H0;!R;D1]=[C;H0;!R;D3][N;H0;!R;D3]([C;H2;!R;D2])[C;H3;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_091(mol):
    """CON(CH2)2"""
    query = Chem.MolFromSmarts('[O;H0;!R;D1]=[C;H0;!R;D3][N;H0;!R;D3]([C;H2;!R;D2])[C;H2;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_092(mol):
    """CONHCO"""
    query = Chem.MolFromSmarts('[O;H0;!R;D1]=[C;H0;!R;D3][N;H1;!R;D2][C;H0;!R;D3]=[O;H0;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_093(mol):
    """CONCO"""
    query = Chem.MolFromSmarts('[O;H0;!R;D1]=[C;H0;!R;D3][N;H0;!R;D3][C;H0;!R;D3]=[O;H0;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_094(mol):
    """aC-CONH2"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[C;H0;!R;D3](=[O;H0;!R;D1])[N;H2;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_095(mol):
    """aC-NH(CO)H"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[N;H1;!R;D2]-[C;H1;!R;D2](=[O;H0;!R;D1])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_096(mol):
    """aC-N(CO)H"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[N;H0;!R;D3]-[C;H1;!R;D2](=[O;H0;!R;D1])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_097(mol):
    """aC-CONH"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[C;H0;!R;D3](=[O;H0;!R;D1])[N;H1;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_098(mol):
    """aC-NHCO"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[N;H1;!R;D2]-[C;H0;!R;D3]=[O;H0;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_099(mol):
    """aC-NCO"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[N;H0;!R;D3]-[C;H0;!R;D3]=[O;H0;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_100(mol):
    """NHCONH"""
    query = Chem.MolFromSmarts('[N;H1;!R;D2]-[C;H0;!R;D3](=[O;H0;!R;D1])-[N;H1;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_101(mol):
    """NH2CONH"""
    query = Chem.MolFromSmarts('[N;H2;!R;D1]-[C;H0;!R;D3](=[O;H0;!R;D1])-[N;H1;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_102(mol):
    """NH2CON"""
    query = Chem.MolFromSmarts('[N;H2;!R;D1]-[C;H0;!R;D3](=[O;H0;!R;D1])-[N;H0;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_103(mol):
    """NHCON"""
    query = Chem.MolFromSmarts('[N;H1;!R;D2]-[C;H0;!R;D3](=[O;H0;!R;D1])-[N;H0;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_104(mol):
    """NCON"""
    query = Chem.MolFromSmarts('[N;H0;!R;D3]-[C;H0;!R;D3](=[O;H0;!R;D1])-[N;H0;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_105(mol):
    """aC-NHCONH2"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[N;H1;!R;D2]-[C;H0;!R;D3](=[O;H0;!R;D1])-[N;H2;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_106(mol):
    """aC-NHCONH"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[N;H1;!R;D2]-[C;H0;!R;D3](=[O;H0;!R;D1])-[N;H1;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_107(mol):
    """NHCO except as above"""
    query = Chem.MolFromSmarts('[N;H1;!R;D2]-[C;H0;!R;D3](=[O;H0;!R;D1])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_108(mol):
    """CH2Cl"""
    query = Chem.MolFromSmarts('[C;H2;!R;D2]-[Cl]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_109(mol):
    """CHCl"""
    query = Chem.MolFromSmarts('[C;H1;!R][Cl]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_110(mol):
    """CCl"""
    query = Chem.MolFromSmarts('[C;H0;!R][Cl]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_111(mol):
    """CHCl2"""
    query = Chem.MolFromSmarts('[C;H1;!R;D3](-[Cl])-[Cl]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_112(mol):
    """CCl2"""
    query = Chem.MolFromSmarts('[C;H0;!R](-[Cl])-[Cl]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_113(mol):
    """CCl3"""
    query = Chem.MolFromSmarts('[C;H0;!R;D4](-[Cl])(-[Cl])-[Cl]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_114(mol):
    """CH2F"""
    query = Chem.MolFromSmarts('[C;H2;!R;D2]-[F]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_115(mol):
    """CHF"""
    query = Chem.MolFromSmarts('[C;H1;!R][F]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_116(mol):
    """CF"""
    query = Chem.MolFromSmarts('[C;H0;!R][F]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_117(mol):
    """CHF2"""
    query = Chem.MolFromSmarts('[C;H1;!R;D3](-[F])-[F]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_118(mol):
    """CF2"""
    query = Chem.MolFromSmarts('[C;H0;!R](-[F])-[F]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_119(mol):
    """CF3"""
    query = Chem.MolFromSmarts('[C;H0;!R;D4](-[F])(-[F])-[F]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_120(mol):
    """CCl2F"""
    query = Chem.MolFromSmarts('F[C;H0;!R;D4](Cl)Cl')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_121(mol):
    """HCClF"""
    query = Chem.MolFromSmarts('F[C;H1;!R;D3]Cl')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_122(mol):
    """CClF2"""
    query = Chem.MolFromSmarts('Cl[C;H0;!R;D4](F)F')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_123(mol):
    """aC-Cl"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-Cl')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_124(mol):
    """aC-F"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-F')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_125(mol):
    """aC-I"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-I')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_126(mol):
    """aC-Br"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-Br')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_127(mol):
    """I except as above"""
    query = Chem.MolFromSmarts('I')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_128(mol):
    """Br except as above"""
    query = Chem.MolFromSmarts('Br')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_129(mol):
    """F except as above"""
    query = Chem.MolFromSmarts('F')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_130(mol):
    """Cl except as above"""
    query = Chem.MolFromSmarts('Cl')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_131(mol):
    """CHNOH"""
    query = Chem.MolFromSmarts('[C;H1;!R;D2]=[N;H0;!R;D2]-[O;H1;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_132(mol):
    """CNOH"""
    query = Chem.MolFromSmarts('[C;H0;!R]=[N;H0;!R;D2]-[O;H1;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_133(mol):
    """aC-CHNOH"""
    query = Chem.MolFromSmarts('[c;H0;R;D3][C;H1;!R;D2]=[N;H0;!R;D2]-[O;H1;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_134(mol):
    """OCH2CH2OH"""
    query = Chem.MolFromSmarts('[O;H0;!R;D2][C;H2;!R;D2][C;H2;!R;D2][O;H1;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_135(mol):
    """OCHCH2OH"""
    query = Chem.MolFromSmarts('[O;H0;!R;D2][C;H1;!R;D3][C;H2;!R;D2][O;H1;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_136(mol):
    """OCH2CHOH"""
    query = Chem.MolFromSmarts('[O;H0;!R;D2][C;H2;!R;D2][C;H1;!R;D3][O;H1;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_137(mol):
    """O-OH"""
    query = Chem.MolFromSmarts('[O;H0;!R;D2][O;H1;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_138(mol):
    """CH2SH"""
    query = Chem.MolFromSmarts('[C;H2;!R;D2]-[S;H1;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_139(mol):
    """CHSH
    这里要求C是D3，是因为后面又142(-SH(except as above))
    """
    query = Chem.MolFromSmarts('[C;H1;!R;D3]-[S;H1;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_140(mol):
    """CSH
    这里要求C是D4，是因为后面又142(-SH(except as above))
    """
    query = Chem.MolFromSmarts('[C;H0;!R;D4]-[S;H1;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_141(mol):
    """aC-SH"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[S;H1;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_142(mol):
    """SH except as above"""
    query = Chem.MolFromSmarts('[S;H1;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_143(mol):
    """CH3S"""
    query = Chem.MolFromSmarts('[C;H3;!R;D1]-[S;H0;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_144(mol):
    """CH2S"""
    query = Chem.MolFromSmarts('[C;H2;!R;D2]-[S;H0;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_145(mol):
    """CHS"""
    query = Chem.MolFromSmarts('[C;H1;!R]-[S;H0;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_146(mol):
    """CS"""
    query = Chem.MolFromSmarts('[C;H0;!R]-[S;H0;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_147(mol):
    """aC-S-"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[S;H0;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_148(mol):
    """SO"""
    query = Chem.MolFromSmarts('[S;H0;!R;D3]=[O;H0;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_149(mol):
    """SO2"""
    query = Chem.MolFromSmarts('[S;H0;!R;D4](=[O;H0;!R;D1])(=[O;H0;!R;D1])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_150(mol):
    """SO3(sulfite)"""
    query = Chem.MolFromSmarts('[O;H0;!R;D2][S;H0;!R;D3](=[O;H0;!R;D1])[O;H0;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_151(mol):
    """SO3(sulfonate)"""
    query = Chem.MolFromSmarts('[O;H0;!R;D2][S;H0;!R;D4](=[O;H0;!R;D1])(=[O;H0;!R;D1])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_152(mol):
    """SO4(sulfite)"""
    query = Chem.MolFromSmarts('[O;H0;!R;D2][S;H0;!R;D4](=[O;H0;!R;D1])(=[O;H0;!R;D1])[O;H0;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_153(mol):
    """aC-SO"""
    query = Chem.MolFromSmarts('[c;H0;R;D3][S;H0;!R;D3]=[O;H0;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_154(mol):
    """aC-SO2"""
    query = Chem.MolFromSmarts('[c;H0;R;D3][S;H0;!R;D4](=[O;H0;!R;D1])(=[O;H0;!R;D1])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


# todo 含磷化合物太复杂了，以后再检查吧
def f_155(mol):
    """PH(phosphine)"""
    query = Chem.MolFromSmarts('[P;H1;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_156(mol):
    """P(phosphine)"""
    query = Chem.MolFromSmarts('[P;H0;D3]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_157(mol):
    """PO3(phosphine)"""
    query = Chem.MolFromSmarts('[O;D2;H0][P;H0;D3]([O;D2;H0])[O;D2;H0]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_158(mol):
    """PHO3(phosphonate)"""
    query = Chem.MolFromSmarts('[O;D2;H0][P;H1;D3](=[O;D1;H0])[O;D2;H0]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_159(mol):
    """PO3(phosphonate)"""
    query = Chem.MolFromSmarts('[O;D2;H0][P;H0;D4](=[O;D1;H0])[O;D2;H0]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_160(mol):
    """PHO4(phosphate)"""
    query = Chem.MolFromSmarts('[O;D2;H0][P;H0;D4]([O;H1;D1])(=[O;D1;H0])[O;D2;H0]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_161(mol):
    """PO4(phosphate)"""
    query = Chem.MolFromSmarts('[O;D2;H0][P;H0;D4]([O;H0;D2])(=[O;D1;H0])[O;D2;H0]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_162(mol):
    """aC-PO4"""
    query = Chem.MolFromSmarts('[c;H0;R][O;D2;H0][P;H0;D4]([O;H0;D2])(=[O;D1;H0])[O;D2;H0]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_163(mol):
    """aC-P"""
    query = Chem.MolFromSmarts('[c;H0;R][P;H0;D3]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_164(mol):
    """CO3(carbonate)"""
    query = Chem.MolFromSmarts('[O;H0;!R;D2][C;H0;!R;D3](=[O;H0;!R;D1])[O;H0;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_165(mol):
    """C2H3O"""
    query = Chem.MolFromSmarts('[C;H2;R;D2]1[O;H0;R;D2][C;H1;R;D3]1')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_166(mol):
    """C2H2O"""
    query = Chem.MolFromSmarts('[C;H2;R;D2]1[O;H0;R;D2][C;H0;R]1')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_167(mol):
    """C2O"""
    query = Chem.MolFromSmarts('[C;H1;R][O;H0;R][C;H0;R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_168(mol):
    """CH2(cyclic)"""
    query = Chem.MolFromSmarts('[C;H2;R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_169(mol):
    """CH(cyclic)"""
    query = Chem.MolFromSmarts('[C;H1;R;D3]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_170(mol):
    """C(cyclic)"""
    query = Chem.MolFromSmarts('[C;H0;R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_171(mol):
    """CH=CH(cyclic)"""
    query = Chem.MolFromSmarts('[C;H1;D2]=[C;H1;R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_172(mol):
    """CH=C(cyclic)"""
    query = Chem.MolFromSmarts('[C;H1;D2]=[C;H0;R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_173(mol):
    """C=C(cyclic)"""
    query = Chem.MolFromSmarts('[C;H0]=[C;H0;R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_174(mol):
    """CH2=C(cyclic)"""
    query = Chem.MolFromSmarts('[C;H2;D1]=[C;H0;R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_175(mol):
    """NH(cyclic)"""
    query = Chem.MolFromSmarts('[N;H1;D2;R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_176(mol):
    """N(cyclic)"""
    query = Chem.MolFromSmarts('[N;H0;D3;R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_177(mol):  # todo 芳香性
    """CH=N(cyclic)"""
    # '[C,c;H1]=,:[N,n;H0;D2;R]'
    query1 = Chem.MolFromSmarts('[C;H1]=[N;H0;D2;R]')
    match_list = mol.GetSubstructMatches(query1)
    return len(match_list), match_list


def f_178(mol):  # todo 芳香性
    """C=N(cyclic)"""
    query = Chem.MolFromSmarts('[C;H0]=[N;H0;D2;R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_179(mol):
    """O(cyclic)"""
    query = Chem.MolFromSmarts('[O;H0;D2;R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_180(mol):
    """CO(cyclic)"""
    query = Chem.MolFromSmarts('[C;H0;R;D3]=[O;H0;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_181(mol):  # todo 芳香性
    """S(cyclic)"""
    query = Chem.MolFromSmarts('[S,s;H0;D2;R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_182(mol):  # todo 芳香性
    """SO2(cyclic)"""
    query = Chem.MolFromSmarts('[O;H0;D1;!R]=[S,s;H0;D4;R]=[O;H0;D1;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


# todo 183-220都没有官方的例子
def f_183(mol):
    """>NH"""
    query = Chem.MolFromSmarts('[N;H1;D3;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_184(mol):
    """-O-"""
    query = Chem.MolFromSmarts('[O;H0;D2;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_185(mol):
    """-S-"""
    query = Chem.MolFromSmarts('[S;H0;D2;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_186(mol):
    """>CO"""
    query = Chem.MolFromSmarts('[C;H0;D3;!R]=[O;H0;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_187(mol):
    """PO2"""
    query = Chem.MolFromSmarts('[O;H0;!R][P;!R][O;H0;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_188(mol):
    """CH-N"""
    query = Chem.MolFromSmarts('[C;H1;!R]-[N;H0;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_189(mol):
    """SiHO"""
    query = Chem.MolFromSmarts('[Si;H1;!R]=[O;H0;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_190(mol):
    """SiO"""
    query = Chem.MolFromSmarts('[Si;H0;!R]=[O;H0;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_191(mol):
    """SiH2"""
    query = Chem.MolFromSmarts('[Si;H2;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_192(mol):
    """SiH1"""
    query = Chem.MolFromSmarts('[Si;H1;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_193(mol):
    """Si"""
    query = Chem.MolFromSmarts('[Si;H0;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_194(mol):
    """(CH3)3N"""
    query = Chem.MolFromSmarts('[C;H3][N;!R]([C;H3])[C;H3]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_195(mol):
    """N=N"""
    query = Chem.MolFromSmarts('[N;H0;!R]=[N;H0;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_196(mol):
    """Ccyc=N-"""
    query = Chem.MolFromSmarts('[C;H0;R]=[N;H0;D2;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_197(mol):
    """Ccyc=CH-"""
    query = Chem.MolFromSmarts('[C;H0;R]=[C;H1;D2;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_198(mol):
    """Ccyc=NH"""
    query = Chem.MolFromSmarts('[C;H0;R]=[N;H1;D1;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_199(mol):
    """N=O"""
    query = Chem.MolFromSmarts('[N;H0;!R]=[O;H0;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_200(mol):
    """Ccyc=C"""
    query = Chem.MolFromSmarts('[C;H0;R]=[C;H0;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_201(mol):
    """P=O"""
    query = Chem.MolFromSmarts('[P;H0;!R]=[O;H0;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_202(mol):
    """N=N"""
    query = Chem.MolFromSmarts('[N;H0;!R]=[N;H0;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_203(mol):
    """C=NH"""
    query = Chem.MolFromSmarts('[C;H0;!R]=[N;H1;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_204(mol):
    """>C=S"""
    query = Chem.MolFromSmarts('[C;H0;!R;D3]=[S;H0;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_205(mol):
    """aC-CON"""
    query = Chem.MolFromSmarts('[c]!@&-[C](=[O;H0;D1])[N;H0]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_206(mol):
    """aC=O"""
    query = Chem.MolFromSmarts('[c]!@&=[O;H0;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_207(mol):
    """aN-"""
    query = Chem.MolFromSmarts('[n;H0;D3]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_208(mol):
    """-Na"""
    query = Chem.MolFromSmarts('[Na]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_209(mol):
    """-K"""
    query = Chem.MolFromSmarts('[K]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_210(mol):
    """HCONH"""
    query = Chem.MolFromSmarts('[C;H1;!R](=[O;H0;D1])[N;H1;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_211(mol):
    """CHOCH"""
    query = Chem.MolFromSmarts('[C;H1;!R](=[O;H0;D1])[C;H1;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_212(mol):
    """C2O"""
    query = Chem.MolFromSmarts('[C;H0;R]1[O;H0;R][C;H0;R]1')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_213(mol):
    """SiH3"""
    query = Chem.MolFromSmarts('[Si;H3;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_214(mol):  # todo ?
    """SiH2O"""
    query = Chem.MolFromSmarts('[Si;H2;!R][O]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_215(mol):
    """CH=C=CH"""
    query = Chem.MolFromSmarts('[C;H1;!R]=[C;H0;!R]=[C;H1;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_216(mol):
    """CH=C=C"""
    query = Chem.MolFromSmarts('[C;H1;!R]=[C;H0;!R]=[C;H0;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_217(mol):
    """OP(=S)O"""
    query = Chem.MolFromSmarts('[O;H0][P;H0](=[S;H0])[O;H0]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_218(mol):  # todo ?
    """R"""
    query = Chem.MolFromSmarts('[*]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_219(mol):
    """CF2cyc"""
    query = Chem.MolFromSmarts('[F][C;H0;R][F]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def f_220(mol):
    """CFcyc"""
    query = Chem.MolFromSmarts('[F][C;H0;R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


# s order
def s_001(mol):
    """(CH3)2CH"""
    query = Chem.MolFromSmarts('[C;H3;!R;D1][C;H1;!R;D3][C;H3;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_002(mol):
    """(CH3)3C"""
    query = Chem.MolFromSmarts('[C;H3;!R;D1][C;H0;!R;D4]([C;H3;!R;D1])[C;H3;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_003(mol):  # todo 能不能把多余的原子去掉？ 好像没有必要
    """CH(CH3)CH(CH3)"""
    query = Chem.MolFromSmarts('[*]-[C;H1;!R;D3]([C;H3;!R;D1])[C;H1;!R;D3]([C;H3;!R;D1])-[*]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_004(mol):  # todo 能不能把多余的原子去掉
    """CH(CH3)C(CH3)2"""
    query = Chem.MolFromSmarts('[*]-[C;H1;!R;D3]([C;H3;!R;D1])[C;H0;!R;D4]([*])([C;H3;!R;D1])[C;H3;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_005(mol):  # todo 能不能把多余的原子去掉
    """C(CH3)2C(CH3)2"""
    query = Chem.MolFromSmarts('[*]-[C;H0;!R;D4]([C;H3;!R;D1])([C;H3;!R;D1])[C;H0;!R;D4]([*])([C;H3;!R;D1])[C;H3;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_006(mol):
    """CHn=CHm-CHp=CHk k,m,n,p=0,1,2"""
    query = Chem.MolFromSmarts('[C;!R]=[C;!R]-[C;!R]=[C;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_007(mol):
    """CH3-CHm=CHn   m,n=0,1,2"""
    query = Chem.MolFromSmarts('[C;H3;!R;D1]-[C;!R]=[C;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_008(mol):
    """CH2-CHm=CHn   m,n=0,1,2"""
    query = Chem.MolFromSmarts('[C;H2;!R;D2]-[C;!R]=[C;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_009(mol):
    """CHp-CHm=CHn   p=0,1 m,n=0,1,2"""
    query = Chem.MolFromSmarts('[C;!H2;!R]-[C;!R]=[C;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_010(mol):
    """CHCHO or CCHO"""
    query1 = Chem.MolFromSmarts('[C;H1;!R][C;H1;!R;D2]=[O;H0;D1]')
    query2 = Chem.MolFromSmarts('[C;H0;!R][C;H1;!R;D2]=[O;H0;D1]')
    match_list = mol.GetSubstructMatches(query1) + mol.GetSubstructMatches(query2)
    return len(match_list), match_list


def s_011(mol):
    """CH3COCH2"""
    query = Chem.MolFromSmarts('[C;H3;!R;D1]-[C;H0;!R;D3](=[O;H0;!R;D1])-[C;H2;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_012(mol):
    """CH3COCH or CH3COC"""
    query = Chem.MolFromSmarts('[C;H3;!R;D1]-[C;H0;!R;D3](=[O;H0;!R;D1])-[C;H1,H0;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_013(mol):
    """CHCOOH or CCOOH"""
    query = Chem.MolFromSmarts('[O;H1;!R;D1]-[C;H0;!R;D3](=[O;H0;!R;D1])-[C;H1,H0;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_014(mol):
    """CH3COOCH or CH3COOC"""
    query = Chem.MolFromSmarts('[C;H3;!R;D1]-[C;H0;!R;D3](=[O;H0;!R;D1])-[O;H0;!R;D2]-[C;H1,H0;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_015(mol):
    """CO-O-CO"""
    query = Chem.MolFromSmarts('[O;H0;!R;D1]=[C;H0;!R;D3]-[O;H0;!R;D2]-[C;H0;!R;D3]=[O;H0;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_016(mol):
    """CHOH"""
    query = Chem.MolFromSmarts('[C;H1;!R]-[O;H1;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_017(mol):
    """COH"""
    query = Chem.MolFromSmarts('[C;H0;!R]-[O;H1;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_018(mol):
    """CH3COCHnOH   n=0,1,2"""
    query = Chem.MolFromSmarts('[C;H3;!R;D1]-[C;H0;!R;D3](=[O;H0;!R;D1])[C;H0,H1,H2;!R]-[O;H1;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_019(mol):
    """NCCHOH or NCCOH"""
    query = Chem.MolFromSmarts('[N;H0;!R;D1]#[C;H0;!R;D2][C;H0,H1;!R]-[O;H1;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_020(mol):
    """OH-CHn-COO   n=0,1,2"""
    query = Chem.MolFromSmarts('[O;H1;!R;D1]-[C;H0,H1,H2;!R]-[C;H0;!R;D3](=[O;H0;!R;D1])-[O;H0;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_021(mol):
    """CHm(OH)CHn(OH)   m,n=0,1,2"""
    query = Chem.MolFromSmarts('[O;H1;!R;D1]-[C;H0,H1,H2;!R]-[C;H0,H1,H2;!R]-[O;H1;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_022(mol):
    """CHm(OH)CHn(NHp)   m,n,p=0,1,2"""
    query = Chem.MolFromSmarts('[O;H1;!R;D1]-[C;H0,H1,H2;!R]-[C;H0,H1,H2;!R]-[N;H0,H1,H2;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_023(mol):
    """CHm(NH2)CHn(NH2)   m,n=0,1,2"""
    query = Chem.MolFromSmarts('[N;H2;!R;D1]-[C;H0,H1,H2;!R]-[C;H0,H1,H2;!R]-[N;H2;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_024(mol):
    """CHm(NH)CHn(NH2)   m,n=0,1,2"""
    query = Chem.MolFromSmarts('[N;H;!R;D2]-[C;H0,H1,H2;!R]-[C;H0,H1,H2;!R]-[N;H2;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_025(mol):
    """H2NCOCHnCHmCONH2   m,n=0,1,2"""
    query = Chem.MolFromSmarts(
        '[N;H2;!R;D1]-[C;H0;!R;D3](=[O;H0;!R;D1])-[C;H0,H1,H2;!R][C;H0,H1,H2;!R]-[C;H0;!R;D3](=[O;H0;!R;D1])-[N;H2;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_026(mol):
    """CHm(NHn)-COOH   m,n=0,1,2"""
    query = Chem.MolFromSmarts('[N;H0,H1,H2;!R]-[C;H0,H1,H2;!R]-[C;H0;!R;D3](=[O;H0;!R;D1])-[O;H1;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_027(mol):
    """HOOC-CHn-COOH   n=1,2"""
    query = Chem.MolFromSmarts(
        '[C;H1,H2;!R](-[C;H0;!R;D3](=[O;H0;!R;D1])-[O;H1;!R;D1])-[C;H0;!R;D3](=[O;H0;!R;D1])-[O;H1;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_028(mol):
    """HOOC-CHn-CHm-COOH   m,n=1,2"""
    query = Chem.MolFromSmarts(
        '[C;H1,H2;!R](-[C;H0;!R;D3](=[O;H0;!R;D1])-[O;H1;!R;D1])-[C;H1,H2;!R](-[C;H0;!R;D3](=[O;H0;!R;D1])-[O;H1;!R;D1])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_029(mol):
    """HO-CHn-COOH   n=0,1,2"""
    query = Chem.MolFromSmarts('[O;H1;!R;D1][C;H0,H1,H2;!R]-[C;H0;!R;D3](=[O;H0;!R;D1])-[O;H1;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_030(mol):
    """NH2-CHn-CHm-COOH   m,n=1,2"""
    query = Chem.MolFromSmarts('[C;H1,H2;!R]([N;H2;!R;D1])-[C;H1,H2;!R](-[C;H0;!R;D3](=[O;H0;!R;D1])-[O;H1;!R;D1])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_031(mol):
    """CH3-O-CHn-COOH   n=1,2"""
    query = Chem.MolFromSmarts('[C;H3;!R;D1]-[O;H0;!R;D2]-[C;H1,H2;!R](-[C;H0;!R;D3](=[O;H0;!R;D1])-[O;H1;!R;D1])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_032(mol):
    """HS-CH-COOH"""
    query = Chem.MolFromSmarts('[S;H1;!R;D1][C;H1;!R](-[C;H0;!R;D3](=[O;H0;!R;D1])-[O;H1;!R;D1])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_033(mol):
    """HS-CHn-CHm-COOH   m,n=1,2"""
    query = Chem.MolFromSmarts('[C;H1,H2;!R]([S;H1;!R;D1])-[C;H1,H2;!R](-[C;H0;!R;D3](=[O;H0;!R;D1])-[O;H1;!R;D1])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_034(mol):
    """NC-CHn-CHm-CN   m,n=1,2"""
    query = Chem.MolFromSmarts('[C;H1,H2;!R]([C;H0;!R;D2]#[N;H0;!R;D1])-[C;H1,H2;!R]([C;H0;!R;D2]#[N;H0;!R;D1])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_035(mol):
    """OH-CHn-CHm-CN   m,n=1,2"""
    query = Chem.MolFromSmarts('[C;H1,H2;!R]([O;H1;!R;D1])-[C;H1,H2;!R]([C;H0;!R;D2]#[N;H0;!R;D1])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_036(mol):
    """HS-CHn-CHm-SH   m,n=1,2"""
    query = Chem.MolFromSmarts('[C;H1,H2;!R]([S;H1;!R;D1])-[C;H1,H2;!R]([S;H1;!R;D1])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_037(mol):
    """COO-CHn-CHm-OOC   m,n=1,2"""
    query = Chem.MolFromSmarts(
        '[C;H1,H2;!R]([O;H0;!R;D2][C;H0;!R;D3](=[O;H0;!R;D1]))-[C;H1,H2;!R]([O;H0;!R;D2][C;H0;!R;D3](=[O;H0;!R;D1]))')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_038(mol):
    """OOC-CHn-CHm-COO   m,n=1,2"""
    query = Chem.MolFromSmarts(
        '[C;H1,H2;!R]([C;H0;D3;!R](=[O;H0;!R;D1])[O;H0;!R;D2])-[C;H1,H2;!R]([C;H0;!R;D3](=[O;H0;!R;D1])[O;H0;!R;D2])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_039(mol):
    """NC-CHn-COO   n=1,2"""
    query = Chem.MolFromSmarts('[C;H1,H2;!R]([C;H0;!R;D2]#[N;H0;!R;D1])([C;H0;!R;D3](=[O;H0;!R;D1])[O;H0;!R;D2])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_040(mol):
    """COCHnCOO   n=1,2"""
    query = Chem.MolFromSmarts('[C;H1,H2;!R]([C;H0;!R;D3](=[O;H0;!R;D1]))([C;H0;D3;!R](=[O;H0;!R;D1])[O;H0;!R;D2])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_041(mol):
    """CHm-O-CHn=CHp   m,n,p=0,1,2,3"""
    query = Chem.MolFromSmarts('[C;H0,H1,H2,H3;!R]-[O;H0;!R;D2]-[C;H0,H1,H2,H3;!R]=[C;H0,H1,H2,H3;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_042(mol):
    """CHm=CHn-F   m,n=0,1,2"""
    query = Chem.MolFromSmarts('[C;H0,H1,H2;!R]=[C;H0,H1,H2;!R]-F')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_043(mol):
    """CHm=CHn-Br   m,n=0,1,2"""
    query = Chem.MolFromSmarts('[C;H0,H1,H2;!R]=[C;H0,H1,H2;!R]-Br')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_044(mol):
    """CHm=CHn-I   m,n=0,1,2"""
    query = Chem.MolFromSmarts('[C;H0,H1,H2;!R]=[C;H0,H1,H2;!R]-I')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_045(mol):
    """CHm=CHn-Cl   m,n=0,1,2"""
    query = Chem.MolFromSmarts('[C;H0,H1,H2;!R]=[C;H0,H1,H2;!R]-Cl')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_046(mol):
    """CHm=CHn-CN   m,n=0,1,2"""
    query = Chem.MolFromSmarts('[C;H0,H1,H2;!R]=[C;H0,H1,H2;!R]-[C;H0;!R;D2]#[N;H0;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_047(mol):
    """CHm=CHn-COO-CHp   m,n,p=0,1,2,3"""
    query = Chem.MolFromSmarts(
        '[C;H0,H1,H2;!R]=[C;H0,H1,H2;!R]-[C;H0;!R](=[O;H0;D1;!R])[O;H0;!R;D2]-[C;H0,H1,H2,H3;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_048(mol):
    """CHm=CHn-CHO   m,n=0,1,2"""
    query = Chem.MolFromSmarts('[C;H0,H1,H2;!R]=[C;H0,H1,H2;!R]-[C;H1;!R;D2](=[O;H0;!R;D1])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_049(mol):
    """CHm=CHn-COOH   m,n=0,1,2"""
    query = Chem.MolFromSmarts('[C;H0,H1,H2;!R]=[C;H0,H1,H2;!R](-[C;H0;!R;D3](=[O;H0;!R;D1])-[O;H1;!R;D1])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_050(mol):
    """aC-CHn-X   n=1,2"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[C;H1,H2;!R]-[F,Cl,Br,I]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_051(mol):
    """aC-CHn-NHm   n=1,2 m=0,1,2"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[C;H1,H2;!R]-[N;H0,H1,H2;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_052(mol):
    """aC-CHn-O-   n=1,2"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[C;H1,H2;!R]-[O;H0;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_053(mol):
    """aC-CHn-OH   n=1,2"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[C;H1,H2;!R]-[O;H1;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_054(mol):
    """aC-CHn-CN   n=1,2"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[C;H1,H2;!R]-[C;H0;!R;D2]#[N;H0;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_055(mol):
    """aC-CHn-CHO   n=1,2"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[C;H1,H2;!R]-[C;H1;!R;D2](=[O;H0;!R;D1])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_056(mol):
    """aC-CHn-SH   n=1,2"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[C;H1,H2;!R]-[S;H1;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_057(mol):
    """aC-CHn-COOH   n=1,2"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[C;H1,H2;!R](-[C;H0;!R;D3](=[O;H0;!R;D1])-[O;H1;!R;D1])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_058(mol):
    """aC-CHn-CO-   n=1,2"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[C;H1,H2;!R]-[C;H0;!R;D3](=[O;H0;!R;D1])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_059(mol):
    """aC-CHn-S-   n=1,2"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[C;H1,H2;!R]-[S;H0;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_060(mol):
    """aC-CHn-OOC-H   n=1,2"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[C;H1,H2;!R]([O;H0;!R;D2][C;H1;!R;D2](=[O;H0;!R;D1]))')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_061(mol):
    """aC-CHn-NO2   n=1,2"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[C;H1,H2;!R]-[N;+]([O;-])=[O]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_062(mol):
    """aC-CHn-CONH2   n=1,2"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[C;H1,H2;!R]-[C;H0;!R;D3](=[O;H0;!R;D1])[N;H2;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_063(mol):
    """aC-CHn-OOC   n=1,2"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[C;H1,H2;!R]([O;H0;!R;D2][C;H0;!R;D3](=[O;H0;!R;D1]))')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_064(mol):
    """aC-CHn-COO   n=1,2"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[C;H1,H2;!R]([C;H0;!R;D3](=[O;H0;!R;D1])[O;H0;!R;D2])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_065(mol):
    """aC-SO2-OH   n=1,2"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[S;H0;!R;D4](=[O;H0;!R;D1])(=[O;H0;!R;D1])[O;H1;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_066(mol):
    """aC-CH(CH3)2"""
    query = Chem.MolFromSmarts('[c;H0;R;D3][C;H1;!R;D3]([C;H3;!R;D1])([C;H3;!R;D1])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_067(mol):
    """aC-C(CH3)3"""
    query = Chem.MolFromSmarts('[c;H0;R;D3][C;H0;!R;D4]([C;H3;!R;D1])([C;H3;!R;D1])([C;H3;!R;D1])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_068(mol):
    """aC-CF3"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[C;H0;!R;D4](F)(F)F')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_069(mol):  # todo 芳香性
    """(CHn=C)(cyclic)-CHO   n=0,1,2"""
    query = Chem.MolFromSmarts('[C;H0,H1,H2;R]=[C;H0;R;D3]-[C;H1;!R;D2](=[O;H0;!R;D1])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_070(mol):  # todo 芳香性
    """(CHn=C)(cyclic)-COO-CHm   m,n=0,1,2,3"""
    query = Chem.MolFromSmarts('[C;H0,H1,H2;R]=[C;H0;R;D3]-[C;H0;!R;D3](=[O;H0;!R;D1])-[O;H0;!R;D2]-[C;H0,H1,H2,H3;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_071(mol):  # todo 芳香性
    """(CHn=C)(cyclic)-CO-   n=0,1,2"""
    query = Chem.MolFromSmarts('[C;H0,H1,H2;R]=[C;H0;R;D3]-[C;H0;!R;D3](=[O;H0;!R;D1])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_072(mol):  # todo 芳香性
    """(CHn=C)cyc-CH3, n=0,1,2"""
    query = Chem.MolFromSmarts('[C;H0,H1,H2;R]=[C;H0;R;D3]-[C;H3;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_073(mol):  # todo 芳香性
    """(CHn=C)cyc-CH2, n=0,1,2"""
    query = Chem.MolFromSmarts('[C;H0,H1,H2;R]=[C;H0;R;D3]-[C;H2;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_074(mol):  # todo 芳香性
    """(CHn=C)(cyclic)-CN   n=0,1,2"""
    query = Chem.MolFromSmarts('[C;H0,H1,H2;R]=[C;H0;R;D3]-[C;H0;!R;D2]#[N;H0;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_075(mol):  # todo 芳香性
    """(CHn=C)(cyclic)-Cl   n=0,1,2"""
    query = Chem.MolFromSmarts('[C;H0,H1,H2;R]=[C;H0;R;D3]-Cl')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_076(mol):
    """CHcyc-CH3"""
    query = Chem.MolFromSmarts('[C;H1;R;D3][C;H3;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_077(mol):
    """CHcyc-CH2"""
    query = Chem.MolFromSmarts('[C;H1;R;D3][C;H2;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_078(mol):
    """CHcyc-CH"""
    query = Chem.MolFromSmarts('[C;H1;R;D3][C;H1;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_079(mol):
    """CHcyc-C"""
    query = Chem.MolFromSmarts('[C;H1;R;D3][C;H0;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_080(mol):
    """CHcyc-CH=CHn   n=1,2"""
    query = Chem.MolFromSmarts('[C;H1;R;D3][C;H1;!R;D2]=[C;H1,H2;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_081(mol):
    """CHcyc-C=CHn n=1,2"""
    query = Chem.MolFromSmarts('[C;H1;R;D3][C;H0;!R;D3]=[C;H1,H2;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_082(mol):
    """CHcyc-Cl"""
    query = Chem.MolFromSmarts('[C;H1;R;D3]-Cl')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_083(mol):
    """CHcyc-F"""
    query = Chem.MolFromSmarts('[C;H1;R;D3]-F')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_084(mol):
    """CHcyc-OH"""
    query = Chem.MolFromSmarts('[C;H1;R;D3]-[O;H1;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_085(mol):
    """CHcyc-NH2"""
    query = Chem.MolFromSmarts('[C;H1;R;D3]-[N;H2;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_086(mol):
    """CHcyc-NH-CHn   n=0,1,2,3"""
    query = Chem.MolFromSmarts('[C;H1;R;D3]-[N;H1;!R;D2]-[C;H0,H1,H2,H3;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_087(mol):
    """CHcyc-N-CHn   n=0,1,2,3"""
    query = Chem.MolFromSmarts('[C;H1;R;D3]-[N;H0;!R;D3]-[C;H0,H1,H2,H3;!R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_088(mol):
    """CHcyc-SH"""
    query = Chem.MolFromSmarts('[C;H1;R;D3]-[S;H1;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_089(mol):
    """CHcyc-CN"""
    query = Chem.MolFromSmarts('[C;H1;R;D3]-[C;H0;!R;D2]#[N;H0;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_090(mol):
    """CHcyc-COOH"""
    query = Chem.MolFromSmarts('[C;H1;R;D3](-[C;H0;!R;D3](=[O;H0;!R;D1])-[O;H1;!R;D1])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_091(mol):
    """CHcyc-CO-"""
    query = Chem.MolFromSmarts('[C;H1;R;D3][C;H0;!R;D3](=[O;H0;!R;D1])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_092(mol):
    """CHcyc-NO2"""
    query = Chem.MolFromSmarts('[C;H1;R;D3]-[N;+]([O;-])=[O]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_093(mol):
    """CHcyc-S-"""
    query = Chem.MolFromSmarts('[C;H1;R;D3]-[S;H0;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_094(mol):
    """CHcyc-CHO"""
    query = Chem.MolFromSmarts('[C;H1;R;D3]-[C;H1;!R;D2](=[O;H0;!R;D1])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_095(mol):
    """CHcyc-O-"""
    query = Chem.MolFromSmarts('[C;H1;R;D3]-[O;H0;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_096(mol):
    """CHcyc-OOCH"""
    query = Chem.MolFromSmarts('[C;H1;R;D3]([O;H0;!R;D2][C;H1;!R;D2](=[O;H0;!R;D1]))')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_097(mol):
    """CHcyc-COO"""
    query = Chem.MolFromSmarts('[C;H1;R;D3]([C;H0;!R;D3](=[O;H0;!R;D1])[O;H0;!R;D2])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_098(mol):
    """CHcyc-OOC"""
    query = Chem.MolFromSmarts('[C;H1;R;D3]([O;H0;!R;D2][C;H0;!R;D3](=[O;H0;!R;D1]))')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_099(mol):
    """Ccyc-CH3
    环上的C要求D4是因为前面有s72
    """
    query = Chem.MolFromSmarts('[C;H0;R;D4][C;H3;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_100(mol):
    """Ccyc-CH2
    环上的C要求D4是因为前面有s73
    """
    query = Chem.MolFromSmarts('[C;H0;R;D4][C;H2;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_101(mol):
    """Ccyc-OH"""
    query = Chem.MolFromSmarts('[C;H0;R;D4][O;H1;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_102(mol):
    """>Ncyc-CH3"""
    query = Chem.MolFromSmarts('[N,n;H0;D3;R]-[C;H3;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_103(mol):
    """>Ncyc-CH2"""
    query = Chem.MolFromSmarts('[N,n;H0;D3;R]-[C;H2;!R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_104(mol):
    """AROMRINGs1s2"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]([!a;!R])[c;H0;R;D3]([!a;!R])[c;H1;R;D2][c;H1;R;D2][c;H1;R;D2][c;H1;R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_105(mol):
    """AROMRINGs1s3"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]([!a;!R])[c;H1;R;D2][c;H0;R;D3]([!a;!R])[c;H1;R;D2][c;H1;R;D2][c;H1;R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_106(mol):
    """AROMRINGs1s4"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]([!a;!R])[c;H1;R;D2][c;H1;R;D2][c;H0;R;D3]([!a;!R])[c;H1;R;D2][c;H1;R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_107(mol):
    """AROMRINGs1s2s3"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]([!a;!R])[c;H0;R;D3]([!a;!R])[c;H0;R;D3]([!a;!R])[c;H1;R;D2][c;H1;R;D2][c;H1;R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_108(mol):
    """AROMRINGs1s2s4"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]([!a;!R])[c;H0;R;D3]([!a;!R])[c;H1;R;D2][c;H0;R;D3]([!a;!R])[c;H1;R;D2][c;H1;R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_109(mol):
    """AROMRINGs1s3s5"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]([!a;!R])[c;H1;R;D2][c;H0;R;D3]([!a;!R])[c;H1;R;D2][c;H0;R;D3]([!a;!R])[c;H1;R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_110(mol):
    """AROMRINGs1s2s3s4"""
    query = Chem.MolFromSmarts(
        '[c;H0;R;D3]([!a;!R])[c;H0;R;D3]([!a;!R])[c;H0;R;D3]([!a;!R])[c;H0;R;D3]([!a;!R])[c;H1;R;D2][c;H1;R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_111(mol):
    """AROMRINGs1s2s3s5"""
    query = Chem.MolFromSmarts(
        '[c;H0;R;D3]([!a;!R])[c;H0;R;D3]([!a;!R])[c;H0;R;D3]([!a;!R])[c;H1;R;D2][c;H0;R;D3]([!a;!R])[c;H1;R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_112(mol):
    """AROMRINGs1s2s4s5"""
    query = Chem.MolFromSmarts(
        '[c;H0;R;D3]([!a;!R])[c;H0;R;D3]([!a;!R])[c;H1;R;D2][c;H0;R;D3]([!a;!R])[c;H0;R;D3]([!a;!R])[c;H1;R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_113(mol):
    """PYRIDINEs2"""
    query = Chem.MolFromSmarts('[n;H0;R;D2][c;H0;R;D3]([!a;!R])[c;H1;R;D2][c;H1;R;D2][c;H1;R;D2][c;H1;R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_114(mol):
    """PYRIDINEs3"""
    query = Chem.MolFromSmarts('[n;H0;R;D2][c;H1;R;D2][c;H0;R;D3]([!a;!R])[c;H1;R;D2][c;H1;R;D2][c;H1;R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_115(mol):
    """PYRIDINEs4"""
    query = Chem.MolFromSmarts('[n;H0;R;D2][c;H1;R;D2][c;H1;R;D2][c;H0;R;D3]([!a;!R])[c;H1;R;D2][c;H1;R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_116(mol):
    """PYRIDINEs2s3"""
    query = Chem.MolFromSmarts('[n;H0;R;D2][c;H0;R;D3]([!a;!R])[c;H0;R;D3]([!a;!R])[c;H1;R;D2][c;H1;R;D2][c;H1;R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_117(mol):
    """PYRIDINEs2s4"""
    query = Chem.MolFromSmarts('[n;H0;R;D2][c;H0;R;D3]([!a;!R])[c;H1;R;D2][c;H0;R;D3]([!a;!R])[c;H1;R;D2][c;H1;R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_118(mol):
    """PYRIDINEs2s5"""
    query = Chem.MolFromSmarts('[n;H0;R;D2][c;H0;R;D3]([!a;!R])[c;H1;R;D2][c;H1;R;D2][c;H0;R;D3]([!a;!R])[c;H1;R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_119(mol):
    """PYRIDINEs2s6"""
    query = Chem.MolFromSmarts('[n;H0;R;D2][c;H0;R;D3]([!a;!R])[c;H1;R;D2][c;H1;R;D2][c;H1;R;D2][c;H0;R;D3]([!a;!R])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_120(mol):
    """PYRIDINEs3s4"""
    query = Chem.MolFromSmarts('[n;H0;R;D2][c;H1;R;D2][c;H0;R;D3]([!a;!R])[c;H0;R;D3]([!a;!R])[c;H1;R;D2][c;H1;R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_121(mol):
    """PYRIDINEs3s5"""
    query = Chem.MolFromSmarts('[n;H0;R;D2][c;H1;R;D2][c;H0;R;D3]([!a;!R])[c;H1;R;D2][c;H0;R;D3]([!a;!R])[c;H1;R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_122(mol):
    """PYRIDINEs2s3s6"""
    query = Chem.MolFromSmarts('[n;H0;R;D2][c;H0;R;D3]([!a;!R])[c;H0;R;D3]([!a;!R])[c;H1;R;D2][c;H1;R;D2][c;H0;R;D3]([!a;!R])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_123(mol):
    """(CHn=CHm)cyc-COOH"""
    query = Chem.MolFromSmarts('[C;R]@&=[C;R]!@&-[C;H0;!R;D3](=[O;H0;!R;D1])-[O;H1;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_124(mol):
    """AROMRINGs1s2s3s4s5"""
    # query = Chem.MolFromSmarts('[c]1[c](!@&-[*])[c](!@&-[*])[c](!@&-[*])[c](!@&-[*])[c]1(!@&-[*])')
    query = Chem.MolFromSmarts(
        '[c;H0;R;D3]([!a;!R])[c;H0;R;D3]([!a;!R])[c;H0;R;D3]([!a;!R])[c;H0;R;D3]([!a;!R])[c;H0;R;D3]([!a;!R])[c;H1;R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_125(mol):
    """aC-NHCOCH2N"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]!@&-[N;H1]-[C;H0](=[O;H0;D1])[C;H2][N]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_126(mol):
    """(N=C)cyc-CH3"""
    query = Chem.MolFromSmarts('[N;R]@&=[C;R]!@&-[C;H3;!R;D1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_127(mol):
    """aC-CONH(CH2)2N"""
    query = Chem.MolFromSmarts('[c]!@&-[C](=[O;H0;D1])[N;H1][C;H2][C;H2][N]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_128(mol):  # todo 原文128、129、130完全一样，不知道怎么回事
    """aC-SO2NHn   n=0,1,2"""
    query = Chem.MolFromSmarts('[c]!@&-[S](=[O;H0;D1])(=[O;H0;D1])[N;H0,H1,H2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_129(mol):  # todo 原文128、129、130完全一样，不知道怎么回事
    """aC-SO2NHn   n=0,1,2"""
    query = Chem.MolFromSmarts('[c]!@&-[S](=[O;H0;D1])(=[O;H0;D1])[N;H0,H1,H2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def s_130(mol):  # todo 原文128、129、130完全一样，不知道怎么回事
    """aC-SO2NHn   n=0,1,2"""
    query = Chem.MolFromSmarts('[c]!@&-[S](=[O;H0;D1])(=[O;H0;D1])[N;H0,H1,H2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


# t order
def t_001(mol):
    """HOOC-(CHn)m-COOH   m>2,n=0,1,2"""
    not_in_ring_atoms = [atom for atom in mol.GetAtoms() if not atom.IsInRing()]
    match_list = ()
    for i in range(3, len(not_in_ring_atoms) + 1):
        query = Chem.MolFromSmarts(
            '[C;H0;!R;D3](=[O;H0;!R;D1])(-[O;H1;!R;D1]){}-[C;H0;!R;D3](=[O;H0;!R;D1])(-[O;H1;!R;D1])'.format(
                '-[C;H0,H1,H2;!R]' * i))
        match_list += mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_002(mol):
    """NHn-(CHn)m-COOH   m>2,n=0,1,2"""
    not_in_ring_atoms = [atom for atom in mol.GetAtoms() if not atom.IsInRing()]
    match_list = ()
    for i in range(3, len(not_in_ring_atoms) + 1):
        query = Chem.MolFromSmarts(
            '[N;H0,H1,H2;!R]{}-[C;H0;!R;D3](=[O;H0;!R;D1])(-[O;H1;!R;D1])'.format('-[C;H0,H1,H2;!R]' * i))
        match_list += mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_003(mol):
    """NH2-(CHn)m-OH   m>2,n=0,1,2"""
    not_in_ring_atoms = [atom for atom in mol.GetAtoms() if not atom.IsInRing()]
    match_list = ()
    for i in range(3, len(not_in_ring_atoms) + 1):
        query = Chem.MolFromSmarts('[N;H2;!R;D1]{}-[O;H1;!R;D1]'.format('-[C;H0,H1,H2;!R]' * i))
        match_list += mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_004(mol):
    """OH-(CHn)m-OH   m>2,n=0,1,2"""
    not_in_ring_atoms = [atom for atom in mol.GetAtoms() if not atom.IsInRing()]
    match_list = ()
    for i in range(3, len(not_in_ring_atoms) + 1):
        query = Chem.MolFromSmarts('[O;H1;!R;D1]{}-[O;H1;!R;D1]'.format('-[C;H0,H1,H2;!R]' * i))
        match_list += mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_005(mol):
    """OH-(CHp)k-O-(CHn)m-OH   m,k>0;p,n=0,1,2"""
    not_in_ring_atoms = [atom for atom in mol.GetAtoms() if not atom.IsInRing()]
    match_list = ()
    for i in range(1, len(not_in_ring_atoms) + 1):
        for j in range(1, len(not_in_ring_atoms) + 1):
            query = Chem.MolFromSmarts(
                '[O;H1;!R;D1]{}-[O;H0;D2;!R]{}-[O;H1;!R;D1]'.format('-[C;H0,H1,H2;!R]' * i, '-[C;H0,H1,H2;!R]' * j))
            match_list += mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_006(mol):
    """OH-(CHp)k-S-(CHn)m-OH   m,k>0;p,n=0,1,2"""
    not_in_ring_atoms = [atom for atom in mol.GetAtoms() if not atom.IsInRing()]
    match_list = ()
    for i in range(1, len(not_in_ring_atoms) + 1):
        for j in range(1, len(not_in_ring_atoms) + 1):
            query = Chem.MolFromSmarts(
                '[O;H1;!R;D1]{}-[S;H0;D2;!R]{}-[O;H1;!R;D1]'.format('-[C;H0,H1,H2;!R]' * i, '-[C;H0,H1,H2;!R]' * j))
            match_list += mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_007(mol):
    """OH-(CHp)k-NHx-(CHn)m-OH   m,k>0;x,p,n=0,1,2"""
    not_in_ring_atoms = [atom for atom in mol.GetAtoms() if not atom.IsInRing()]
    match_list = ()
    for i in range(1, len(not_in_ring_atoms) + 1):
        for j in range(1, len(not_in_ring_atoms) + 1):
            query = Chem.MolFromSmarts(
                '[O;H1;!R;D1]{}-[N;H0,H1,H2;!R]{}-[O;H1;!R;D1]'.format('-[C;H0,H1,H2;!R]' * i, '-[C;H0,H1,H2;!R]' * j))
            match_list += mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_008(mol):
    """CHp-O-(CHn)m-OH   m>2;p,n=0,1,2"""
    not_in_ring_atoms = [atom for atom in mol.GetAtoms() if not atom.IsInRing()]
    match_list = ()
    for i in range(3, len(not_in_ring_atoms) + 1):
        query = Chem.MolFromSmarts('[C;H0,H1,H2;!R][O;H0;!R;D2]{}-[O;H1;!R;D1]'.format('-[C;H0,H1,H2;!R]' * i))
        match_list += mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_009(mol):
    """NH2-(CHn)m-NH2   m>2,n=0,1,2"""
    not_in_ring_atoms = [atom for atom in mol.GetAtoms() if not atom.IsInRing()]
    match_list = ()
    for i in range(3, len(not_in_ring_atoms) + 1):
        query = Chem.MolFromSmarts('[N;H2;!R;D1]{}-[N;H2;!R;D1]'.format('-[C;H0,H1,H2;!R]' * i))
        match_list += mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_010(mol):
    """NHk-(CHn)m-NH2   m>2;n=0,1,2;k=0,1"""
    not_in_ring_atoms = [atom for atom in mol.GetAtoms() if not atom.IsInRing()]
    match_list = ()
    for i in range(3, len(not_in_ring_atoms) + 1):
        query = Chem.MolFromSmarts('[N;H0,H1;!R]{}-[N;H2;!R;D1]'.format('-[C;H0,H1,H2;!R]' * i))
        match_list += mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_011(mol):
    """SH-(CHn)m-SH   m>2;n=0,1,2"""
    not_in_ring_atoms = [atom for atom in mol.GetAtoms() if not atom.IsInRing()]
    match_list = ()
    for i in range(3, len(not_in_ring_atoms) + 1):
        query = Chem.MolFromSmarts('[S;H1;!R;D1]{}-[S;H1;!R;D1]'.format('-[C;H0,H1,H2;!R]' * i))
        match_list += mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_012(mol):
    """CN-(CHn)m-CN   m>2;n=0,1,2"""
    not_in_ring_atoms = [atom for atom in mol.GetAtoms() if not atom.IsInRing()]
    match_list = ()
    for i in range(3, len(not_in_ring_atoms) + 1):
        query = Chem.MolFromSmarts('[C;H0;!R;D2](#[N;H0;!R;D1]){}-[C;H0;!R;D2](#[N;H0;!R;D1])'.format('-[C;H0,H1,H2;!R]' * i))
        match_list += mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_013(mol):
    """COO-(CHn)m-OOC   m>2;n=0,1,2"""
    not_in_ring_atoms = [atom for atom in mol.GetAtoms() if not atom.IsInRing()]
    match_list = ()
    for i in range(3, len(not_in_ring_atoms) + 1):
        query = Chem.MolFromSmarts(
            '[C;H0;!R;D3](=[O;H0;!R;D1])-[O;H0;!R;D2]{}-[O;H0;!R;D2]-[C;H0;!R;D3](=[O;H0;!R;D1])'.format(
                '-[C;H0,H1,H2;!R]' * i))
        match_list += mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_014(mol):
    """aC-(CHn=CHm)cyc (fused rings)   m,n=0,1"""
    query = Chem.MolFromSmarts('[c;H0;R2;D3]@&-[C;H1,H2;R]@&=[C;H1,H2;R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_015(mol):
    """aC-aC (different rings)"""
    query = Chem.MolFromSmarts('[c;H0;R1;D3]!@&-[c;H0;R1;D3]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_016(mol):
    """aC-CHncyc (different rings)   n=0,1"""
    query = Chem.MolFromSmarts('[c;H0;R1;D3]!@&-[C;H0,H1;R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_017(mol):
    """aC-CHncyc (fused rings)   n=0,1,2原文写的0,1应该是不对的"""
    query = Chem.MolFromSmarts('[c;H0;R2;D3]@&-[C;H0,H1,H2;R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_018(mol):
    """aC-(CHn)m-aC m>1 n=0,1,2"""
    not_in_ring_atoms = [atom for atom in mol.GetAtoms() if not atom.IsInRing()]
    match_list = ()
    for i in range(2, len(not_in_ring_atoms) + 1):
        query = Chem.MolFromSmarts('[c;H0;R;D3]{}-[c;H0;R;D3]'.format('-[C;!R]' * i))
        match_list += mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_019(mol):
    """aC-(CHn)m-CHcyc m>1 n=0,1,2"""
    not_in_ring_atoms = [atom for atom in mol.GetAtoms() if not atom.IsInRing()]
    match_list = ()
    for i in range(1, len(not_in_ring_atoms) + 1):
        query = Chem.MolFromSmarts('[c;H0;R;D3]{}-[C;H1;R;D3]'.format('-[C;!R]' * i))
        match_list += mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_020(mol):
    """CHcyc-CHcyc (different rings)"""
    query = Chem.MolFromSmarts('[C;H1;R;D3]-&!@[C;H1;R;D3]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_021(mol):
    """CHcyc-(CHn)m-CHcyc (different rings)   m>0, n=0,1,2"""
    not_in_ring_atoms = [atom for atom in mol.GetAtoms() if not atom.IsInRing()]
    match_list = ()
    for i in range(1, len(not_in_ring_atoms) + 1):
        query = Chem.MolFromSmarts('[C;H1;R;D3]{}!@&-[C;H1;R;D3]'.format('!@&-[C;H0,H1,H2;!R]' * i))
        match_list += mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_022(mol):
    """CH multiring"""
    query = Chem.MolFromSmarts('[C;H1;!R1&R;D3]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_023(mol):
    """C multiring"""
    query = Chem.MolFromSmarts('[C;H0;!R1&R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_024(mol):
    """aC-CHm-aC (different rings)   m=0,1,2"""
    query = Chem.MolFromSmarts('[c;H0;R1]-[C;H0,H1,H2;!R]-[c;H0;R1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_025(mol):
    """aC-(CHm=CHn)-aC (different rings)   m,n=0,1,2"""
    query = Chem.MolFromSmarts('[c;H0;R1;D3]-[C;H0,H1,H2;!R]=[C;H0,H1,H2;!R]-[c;H0;R1;D3]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_026(mol):  # todo 芳香性
    """(CHm=C)cyc-CH=CH-(C=CHn)cyc (different rings)   m,n没有限制"""
    query = Chem.MolFromSmarts('[C;R]=[C;H0;R]-[C;H1;!R]=[C;H1;!R]-[C;H0;R]=[C;R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_027(mol):  # todo 芳香性
    """(CHm=C)cyc-CHp-(C=CHn)cyc (different rings)   m,n,p没有限制"""
    query = Chem.MolFromSmarts('[C;R]=[C;H0;R]-[C;!R]-[C;H0;R]=[C;R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_028(mol):
    """aC-CO-aC (different rings)"""
    query = Chem.MolFromSmarts('[c;H0;R1;D3]-[C;H0;!R;D3](=[O;H0;!R;D1])-[c;H0;R1;D3]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_029(mol):
    """aC-CHm-CO-aC (different rings)   m=0,1,2"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[C;H0,H1,H2;!R]-[C;H0;!R;D3](=[O;H0;!R;D1])-[c;H0;R;D3]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_030(mol):  # todo 芳香性
    """aC-CO-(C=CHn)cyc (different rings)"""
    query = Chem.MolFromSmarts('[c;H0;R1;D3]-[C;H0;!R;D3](=[O;H0;!R;D1])-[C;H0;R;D3]=[C;H0,H1;R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_031(mol):
    """aC-CO-CO-aC (different rings)"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[C;H0;!R;D3](=[O;H0;!R;D1])-[C;H0;!R;D3](=[O;H0;!R;D1])-[c;H0;R;D3]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_032(mol):
    """aC-COcyc (fused rings)"""
    query = Chem.MolFromSmarts('[c;H0;R2;D3]@&-[C;R;H0;D3](=[O;H0;!R;D1])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_033(mol):
    """aC-CO-(CHn)m-CO-aC (different rings)   m>0;n=0,1,2"""
    not_in_ring_atoms = [atom for atom in mol.GetAtoms() if not atom.IsInRing()]
    match_list = ()
    for i in range(1, len(not_in_ring_atoms) + 1):
        query = Chem.MolFromSmarts(
            '[c;H0;R;D3]-[C;H0;!R;D3](=[O;H0;!R;D1]){}-[C;H0;!R;D3](=[O;H0;!R;D1])-[c;H0;R;D3]'.format('-[C;H0,H1,H2;!R]' * i))
        match_list += mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_034(mol):
    """aC-CO-CHn,cyc (different rings)   n=0,1"""
    query = Chem.MolFromSmarts('[c;H0;R1;D3]-[C;H0;!R;D3](=[O;H0;!R;D1])-[C;H0,H1;R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_035(mol):
    """aC-CO-NHn-aC (different rings)   n=0,1"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[C;H0;!R;D3](=[O;H0;!R;D1])-[N;H0,H1;!R]-[c;H0;R;D3]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_036(mol):
    """aC-NHnCONHm-aC (different rings)   m,n=0,1"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[N;H0,H1;!R]-[C;H0;!R;D3](=[O;H0;!R;D1])-[N;H0,H1;!R]-[c;H0;R;D3]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_037(mol):
    """aC-CO-Ncyc (different rings)"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]!@&-[C;H0;!R;D3](=[O;H0;!R;D1])!@&-[N;H0;R;D3]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_038(mol):
    """aC-Scyc (fused rings)"""
    query = Chem.MolFromSmarts('[c;H0;R2;D3]@[S,s;H0;D2;R1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_039(mol):
    """aC-S-aC (different rings)"""
    query = Chem.MolFromSmarts('[c;H0;R1;D3]!@&-[S;H0;D2;!R]!@&-[c;H0;R1;D3]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_040(mol):  # todo  ?
    """aC-POn-aC (different rings)   n=0,1,2,3,4"""
    query0 = Chem.MolFromSmarts('[c;H0;R1;D3]-[P;H0,H1;!R]-[c;H0;R1;D3]')
    query1 = Chem.MolFromSmarts('[c;H0;R1;D3]-[O;H0;D2;!R]-[P;!R]-[O;H0;D2;!R]-[c;H0;R1;D3]')
    match_list = mol.GetSubstructMatches(query0) + mol.GetSubstructMatches(query1)
    return len(match_list), match_list


def t_041(mol):
    """aC-SOn-aC (different rings)   n=1,2,3,4"""
    query1 = Chem.MolFromSmarts('[c;H0;R1;D3]-[S;H0;D3;!R](=[O;!R;H0;D1])-[c;H0;R1;D3]')
    query2 = Chem.MolFromSmarts('[c;H0;R1;D3]-[S;H0;D4;!R](=[O;!R;H0;D1])(=[O;!R;H0;D1])-[c;H0;R1;D3]')
    query3 = Chem.MolFromSmarts('[c;H0;R1;D3]-[O;H0;D2;!R]-[S;H0;D3;!R](=[O;!R;H0;D1])-[O;H0;D2;!R]-[c;H0;R1;D3]')
    query4 = Chem.MolFromSmarts(
        '[c;H0;R1;D3]-[O;H0;D2;!R]-[S;H0;D4;!R](=[O;!R;H0;D1])(=[O;!R;H0;D1])-[O;H0;D2;!R]-[c;H0;R1;D3]')
    match_list = mol.GetSubstructMatches(query1) + mol.GetSubstructMatches(query2) + mol.GetSubstructMatches(
        query3) + mol.GetSubstructMatches(query4)
    return len(match_list), match_list


def t_042(mol):
    """aC-NHncyc (fused rings)   n=0,1"""
    query = Chem.MolFromSmarts('[c;H0;R2;D3]@[N,n;H0,H1;R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_043(mol):
    """aC-NH-aC (different rings)"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[N;H1;!R;D2]-[c;H0;R;D3]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_044(mol):
    """aC-(C=N)cyc (different rings)"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]!@&-[c,C;H0;R;D3]=,:[n,N;H0;R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_045(mol):
    """aC-(N=CHn)cyc (fused rings)   n=0,1"""
    # query = Chem.MolFromSmarts('[c][c;H0;R2]([c])-[N;H0;D2;R1]=[C;H0,H1;R]')
    query = Chem.MolFromSmarts('[c;H0;R;D3]@[n,N;H0;R;D2]@,=,:[c,C;R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_046(mol):
    """aC-(N=CHn)cyc (fused rings)   n=0,1"""
    query = Chem.MolFromSmarts('[c;H0;R2;D3]@,=,:[c,C;H0,H1;R]@,=,:[n,N;H0;D2;R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_047(mol):
    """aC-O-CHn-aC (different rings)   n=0,1,2"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]!@,-[O;H0;!R;D2]!@,-[C;H0,H1,H2;!R]!@,-[c;H0;R;D3]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_048(mol):
    """aC-O-aC (different rings)"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[O;H0;!R;D2]-[c;H0;R;D3]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_049(mol):
    """aC-CHn-O-CHm-aC (different rings)   m,n=0,1,2"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]-[C;H0,H1,H2;!R]-[O;H0;!R;D2]-[C;H0,H1,H2;!R]-[c;H0;R;D3]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_050(mol):
    """aC-Ocyc (fused rings)"""
    query = Chem.MolFromSmarts('[c;H0;R2;D3]@[O,o;H0;R;D2]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_051(mol):
    """AROMFUSED[2]"""
    query = Chem.MolFromSmarts('[c]:[c;R2](:[c]):[c]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_052(mol):
    """AROMFUSED[2]s1"""
    query = Chem.MolFromSmarts('[c;R2]([c;H1]):[c;R2]([c;H1])c([C,O,N,P,S,F,Cl,Br,I])[c;H1][c;H1][c;H1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_053(mol):
    """AROMFUSED[2]s2"""
    query = Chem.MolFromSmarts('[*]!@[c]1[c;H1][c;R2]([c])[c;R2]([c])[c;H1][c;H1]1')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_054(mol):
    """AROMFUSED[2]s2s3"""
    query = Chem.MolFromSmarts('[*]!@[c]1[c](!@[*])[c;H1][c;R2]([c])[c;R2]([c])[c;H1]1')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_055(mol):
    """AROMFUSED[2]s1s4"""
    query = Chem.MolFromSmarts('[c;H1]1[c](!@[*])[c;R2]([c])[c;R2]([c])[c](!@[*])[c;H1]1')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_056(mol):
    """AROMFUSED[2]s1s2"""
    query = Chem.MolFromSmarts('[c;H1]1[c](!@[*])[c](!@[*])[c;R2]([c])[c;R2]([c])[c;H1]1')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_057(mol):
    """AROMFUSED[2]s1s3"""
    query = Chem.MolFromSmarts('[c;H1]1[c](!@[*])[c;H1][c;R2]([c])[c;R2]([c])[c](!@[*])1')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_058(mol):
    """AROMFUSED[3]"""
    query = Chem.MolFromSmarts('[c][c;R2]1[c;R3]([c])[c;R2]([c])ccc1')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_059(mol):
    """AROMFUSED[4a]"""
    query = Chem.MolFromSmarts('[c;H1][c;R2]1[c;R2]([c])[c;H1][c;R2]([c])[c;R2]([c])[c;H1]1')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_060(mol):
    """AROMFUSED[4a]s1"""
    query = Chem.MolFromSmarts('[*]!@[c]1[c;R2]([c])[c;R2]([c])[c;H1][c;R2]([c])[c;R2]1([c])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_061(mol):
    """AROMFUSED[4a]s1s4"""
    query = Chem.MolFromSmarts('[*]!@[c]1[c;R2]([c])[c;R2]([c])[c](!@[*])[c;R2]([c])[c;R2]1([c])')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_062(mol):
    """AROMFUSED[4p]"""
    query = Chem.MolFromSmarts('[c;H1]1[c;!R1]([c])[c;!R1]([c])[c;!R1]([c])[c;!R1]([c])[c;H1]1')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_063(mol):
    """AROMFUSED[4p]s3s4"""
    query = Chem.MolFromSmarts('[*]!@[c]1[c;!R1]([c])[c;!R1]([c])[c;!R1]([c])[c;!R1]([c])[c]1!@[*]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_064(mol):
    """PYRIDINE.FUSED[2]"""
    query = Chem.MolFromSmarts('[n;H0;R1;D2]1[c;H1;R1;D2][c;H1;R1;D2][c;H1;R1;D2][c;H0;R2;D3](a)[c;H0;R2;D3]1(a)')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_065(mol):
    """PYRIDINE.FUSED[2-iso]"""
    query = Chem.MolFromSmarts('[n;H0;R1;D2]1[c;H1;R1;D2][c;H1;R1;D2][c;H0;R2;D3](a)[c;H0;R2;D3](a)[c;H1;R1;D2]1')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_066(mol):
    """PYRIDINE.FUSED[4]"""
    query = Chem.MolFromSmarts('[n;H0;R1;D2]1[c;H0;R2;D3](a)[c;H0;R2;D3](a)[c;H1;R1;D2][c;H0;R2;D3](a)[c;H0;R2;D3]1(a)')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_067(mol):
    """aC-N-CHcyc(different rings)"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]!@&-[N]!@&-[C;H1;R]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_068(mol):
    """N multiring"""
    query = Chem.MolFromSmarts('[N;R&!R1]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_069(mol):
    """Ncyc-(CH2)3-Ncyc(different rings)"""
    query = Chem.MolFromSmarts('[N;H0;R;D3]!@&-[C;H2][C;H2][C;H2]!@&-[N;H0;R;D3]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_070(mol):
    """aC-COCH2CH2-aC(different rings)"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]!@&-[C;H0](=[O;H0;!R;D1])[C;H2][C;H2]!@&-[c;H0;R;D3]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_071(mol):
    """aC-O-(CH2)2-Ncyc(different rings)"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]!@&-[O;H0;D2]-[C;H2][C;H2]!@&-[N;H0;R;D3]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_072(mol):
    """aC-CH(OH)(CH2)2-CHcyc(different rings)"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]!@&-[C;H1]([O;H1;!R;D1])-[C;H2][C;H2]!@&-[C;H1;R;D3]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_073(mol):
    """Ncyc-(CH2)2-CHcyc(different rings)"""
    query = Chem.MolFromSmarts('[N;H0;R;D3]!@&-[C;H2][C;H2]!@&-[C;H1;R;D3]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


def t_074(mol):
    """aC-CONHCH2-CHcyc(different rings)"""
    query = Chem.MolFromSmarts('[c;H0;R;D3]!@&-[C;H0;D3](=[O;H0;!R;D1])[N;H1;D2][C;H2]!@&-[C;H1;R;D3]')
    match_list = mol.GetSubstructMatches(query)
    return len(match_list), match_list


class Counter:
    def __init__(self):

        self.init_result = {
            'f_001': 0, 'f_002': 0, 'f_003': 0, 'f_004': 0, 'f_005': 0, 'f_006': 0, 'f_007': 0, 'f_008': 0, 'f_009': 0,
            'f_010': 0, 'f_011': 0, 'f_012': 0, 'f_013': 0, 'f_014': 0, 'f_015': 0, 'f_016': 0, 'f_017': 0, 'f_018': 0,
            'f_019': 0, 'f_020': 0, 'f_021': 0, 'f_022': 0, 'f_023': 0, 'f_024': 0, 'f_025': 0, 'f_026': 0, 'f_027': 0,
            'f_028': 0, 'f_029': 0, 'f_030': 0, 'f_031': 0, 'f_032': 0, 'f_033': 0, 'f_034': 0, 'f_035': 0, 'f_036': 0,
            'f_037': 0, 'f_038': 0, 'f_039': 0, 'f_040': 0, 'f_041': 0, 'f_042': 0, 'f_043': 0, 'f_044': 0, 'f_045': 0,
            'f_046': 0, 'f_047': 0, 'f_048': 0, 'f_049': 0, 'f_050': 0, 'f_051': 0, 'f_052': 0, 'f_053': 0, 'f_054': 0,
            'f_055': 0, 'f_056': 0, 'f_057': 0, 'f_058': 0, 'f_059': 0, 'f_060': 0, 'f_061': 0, 'f_062': 0, 'f_063': 0,
            'f_064': 0, 'f_065': 0, 'f_066': 0, 'f_067': 0, 'f_068': 0, 'f_069': 0, 'f_070': 0, 'f_071': 0, 'f_072': 0,
            'f_073': 0, 'f_074': 0, 'f_075': 0, 'f_076': 0, 'f_077': 0, 'f_078': 0, 'f_079': 0, 'f_080': 0, 'f_081': 0,
            'f_082': 0, 'f_083': 0, 'f_084': 0, 'f_085': 0, 'f_086': 0, 'f_087': 0, 'f_088': 0, 'f_089': 0, 'f_090': 0,
            'f_091': 0, 'f_092': 0, 'f_093': 0, 'f_094': 0, 'f_095': 0, 'f_096': 0, 'f_097': 0, 'f_098': 0, 'f_099': 0,
            'f_100': 0, 'f_101': 0, 'f_102': 0, 'f_103': 0, 'f_104': 0, 'f_105': 0, 'f_106': 0, 'f_107': 0, 'f_108': 0,
            'f_109': 0, 'f_110': 0, 'f_111': 0, 'f_112': 0, 'f_113': 0, 'f_114': 0, 'f_115': 0, 'f_116': 0, 'f_117': 0,
            'f_118': 0, 'f_119': 0, 'f_120': 0, 'f_121': 0, 'f_122': 0, 'f_123': 0, 'f_124': 0, 'f_125': 0, 'f_126': 0,
            'f_127': 0, 'f_128': 0, 'f_129': 0, 'f_130': 0, 'f_131': 0, 'f_132': 0, 'f_133': 0, 'f_134': 0, 'f_135': 0,
            'f_136': 0, 'f_137': 0, 'f_138': 0, 'f_139': 0, 'f_140': 0, 'f_141': 0, 'f_142': 0, 'f_143': 0, 'f_144': 0,
            'f_145': 0, 'f_146': 0, 'f_147': 0, 'f_148': 0, 'f_149': 0, 'f_150': 0, 'f_151': 0, 'f_152': 0, 'f_153': 0,
            'f_154': 0, 'f_155': 0, 'f_156': 0, 'f_157': 0, 'f_158': 0, 'f_159': 0, 'f_160': 0, 'f_161': 0, 'f_162': 0,
            'f_163': 0, 'f_164': 0, 'f_165': 0, 'f_166': 0, 'f_167': 0, 'f_168': 0, 'f_169': 0, 'f_170': 0, 'f_171': 0,
            'f_172': 0, 'f_173': 0, 'f_174': 0, 'f_175': 0, 'f_176': 0, 'f_177': 0, 'f_178': 0, 'f_179': 0, 'f_180': 0,
            'f_181': 0, 'f_182': 0, 'f_183': 0, 'f_184': 0, 'f_185': 0, 'f_186': 0, 'f_187': 0, 'f_188': 0, 'f_189': 0,
            'f_190': 0, 'f_191': 0, 'f_192': 0, 'f_193': 0, 'f_194': 0, 'f_195': 0, 'f_196': 0, 'f_197': 0, 'f_198': 0,
            'f_199': 0, 'f_200': 0, 'f_201': 0, 'f_202': 0, 'f_203': 0, 'f_204': 0, 'f_205': 0, 'f_206': 0, 'f_207': 0,
            'f_208': 0, 'f_209': 0, 'f_210': 0, 'f_211': 0, 'f_212': 0, 'f_213': 0, 'f_214': 0, 'f_215': 0, 'f_216': 0,
            'f_217': 0, 'f_218': 0, 'f_219': 0, 'f_220': 0,

            's_001': 0, 's_002': 0, 's_003': 0, 's_004': 0, 's_005': 0, 's_006': 0, 's_007': 0, 's_008': 0, 's_009': 0,
            's_010': 0, 's_011': 0, 's_012': 0, 's_013': 0, 's_014': 0, 's_015': 0, 's_016': 0, 's_017': 0, 's_018': 0,
            's_019': 0, 's_020': 0, 's_021': 0, 's_022': 0, 's_023': 0, 's_024': 0, 's_025': 0, 's_026': 0, 's_027': 0,
            's_028': 0, 's_029': 0, 's_030': 0, 's_031': 0, 's_032': 0, 's_033': 0, 's_034': 0, 's_035': 0, 's_036': 0,
            's_037': 0, 's_038': 0, 's_039': 0, 's_040': 0, 's_041': 0, 's_042': 0, 's_043': 0, 's_044': 0, 's_045': 0,
            's_046': 0, 's_047': 0, 's_048': 0, 's_049': 0, 's_050': 0, 's_051': 0, 's_052': 0, 's_053': 0, 's_054': 0,
            's_055': 0, 's_056': 0, 's_057': 0, 's_058': 0, 's_059': 0, 's_060': 0, 's_061': 0, 's_062': 0, 's_063': 0,
            's_064': 0, 's_065': 0, 's_066': 0, 's_067': 0, 's_068': 0, 's_069': 0, 's_070': 0, 's_071': 0, 's_072': 0,
            's_073': 0, 's_074': 0, 's_075': 0, 's_076': 0, 's_077': 0, 's_078': 0, 's_079': 0, 's_080': 0, 's_081': 0,
            's_082': 0, 's_083': 0, 's_084': 0, 's_085': 0, 's_086': 0, 's_087': 0, 's_088': 0, 's_089': 0, 's_090': 0,
            's_091': 0, 's_092': 0, 's_093': 0, 's_094': 0, 's_095': 0, 's_096': 0, 's_097': 0, 's_098': 0, 's_099': 0,
            's_100': 0, 's_101': 0, 's_102': 0, 's_103': 0, 's_104': 0, 's_105': 0, 's_106': 0, 's_107': 0, 's_108': 0,
            's_109': 0, 's_110': 0, 's_111': 0, 's_112': 0, 's_113': 0, 's_114': 0, 's_115': 0, 's_116': 0, 's_117': 0,
            's_118': 0, 's_119': 0, 's_120': 0, 's_121': 0, 's_122': 0, 's_123': 0, 's_124': 0, 's_125': 0, 's_126': 0,
            's_127': 0, 's_128': 0, 's_129': 0, 's_130': 0,

            't_001': 0, 't_002': 0, 't_003': 0, 't_004': 0, 't_005': 0, 't_006': 0, 't_007': 0, 't_008': 0, 't_009': 0,
            't_010': 0, 't_011': 0, 't_012': 0, 't_013': 0, 't_014': 0, 't_015': 0, 't_016': 0, 't_017': 0, 't_018': 0,
            't_019': 0, 't_020': 0, 't_021': 0, 't_022': 0, 't_023': 0, 't_024': 0, 't_025': 0, 't_026': 0, 't_027': 0,
            't_028': 0, 't_029': 0, 't_030': 0, 't_031': 0, 't_032': 0, 't_033': 0, 't_034': 0, 't_035': 0, 't_036': 0,
            't_037': 0, 't_038': 0, 't_039': 0, 't_040': 0, 't_041': 0, 't_042': 0, 't_043': 0, 't_044': 0, 't_045': 0,
            't_046': 0, 't_047': 0, 't_048': 0, 't_049': 0, 't_050': 0, 't_051': 0, 't_052': 0, 't_053': 0, 't_054': 0,
            't_055': 0, 't_056': 0, 't_057': 0, 't_058': 0, 't_059': 0, 't_060': 0, 't_061': 0, 't_062': 0, 't_063': 0,
            't_064': 0, 't_065': 0, 't_066': 0, 't_067': 0, 't_068': 0, 't_069': 0, 't_070': 0, 't_071': 0, 't_072': 0,
            't_073': 0, 't_074': 0,
        }
        self.result = None

        self.f_order_group_function = \
            [
                f_001, f_002, f_003, f_004, f_005, f_006, f_007, f_008, f_009, f_010, f_011, f_012, f_013, f_014, f_015,
                f_016, f_017, f_018, f_019, f_020, f_021, f_022, f_023, f_024, f_025, f_026, f_027, f_028, f_029, f_030,
                f_031, f_032, f_033, f_034, f_035, f_036, f_037, f_038, f_039, f_040, f_041, f_042, f_043, f_044, f_045,
                f_046, f_047, f_048, f_049, f_050, f_051, f_052, f_053, f_054, f_055, f_056, f_057, f_058, f_059, f_060,
                f_061, f_062, f_063, f_064, f_065, f_066, f_067, f_068, f_069, f_070, f_071, f_072, f_073, f_074, f_075,
                f_076, f_077, f_078, f_079, f_080, f_081, f_082, f_083, f_084, f_085, f_086, f_087, f_088, f_089, f_090,
                f_091, f_092, f_093, f_094, f_095, f_096, f_097, f_098, f_099, f_100, f_101, f_102, f_103, f_104, f_105,
                f_106, f_107, f_108, f_109, f_110, f_111, f_112, f_113, f_114, f_115, f_116, f_117, f_118, f_119, f_120,
                f_121, f_122, f_123, f_124, f_125, f_126, f_127, f_128, f_129, f_130, f_131, f_132, f_133, f_134, f_135,
                f_136, f_137, f_138, f_139, f_140, f_141, f_142, f_143, f_144, f_145, f_146, f_147, f_148, f_149, f_150,
                f_151, f_152, f_153, f_154, f_155, f_156, f_157, f_158, f_159, f_160, f_161, f_162, f_163, f_164, f_165,
                f_166, f_167, f_168, f_169, f_170, f_171, f_172, f_173, f_174, f_175, f_176, f_177, f_178, f_179, f_180,
                f_181, f_182, f_183, f_184, f_185, f_186, f_187, f_188, f_189, f_190, f_191, f_192, f_193, f_194, f_195,
                f_196, f_197, f_198, f_199, f_200, f_201, f_202, f_203, f_204, f_205, f_206, f_207, f_208, f_209, f_210,
                f_211, f_212, f_213, f_214, f_215, f_216, f_217, f_218, f_219, f_220,
            ]
        self.s_order_group_function = \
            [
                s_001, s_002, s_003, s_004, s_005, s_006, s_007, s_008, s_009, s_010, s_011, s_012, s_013, s_014, s_015,
                s_016, s_017, s_018, s_019, s_020, s_021, s_022, s_023, s_024, s_025, s_026, s_027, s_028, s_029, s_030,
                s_031, s_032, s_033, s_034, s_035, s_036, s_037, s_038, s_039, s_040, s_041, s_042, s_043, s_044, s_045,
                s_046, s_047, s_048, s_049, s_050, s_051, s_052, s_053, s_054, s_055, s_056, s_057, s_058, s_059, s_060,
                s_061, s_062, s_063, s_064, s_065, s_066, s_067, s_068, s_069, s_070, s_071, s_072, s_073, s_074, s_075,
                s_076, s_077, s_078, s_079, s_080, s_081, s_082, s_083, s_084, s_085, s_086, s_087, s_088, s_089, s_090,
                s_091, s_092, s_093, s_094, s_095, s_096, s_097, s_098, s_099, s_100, s_101, s_102, s_103, s_104, s_105,
                s_106, s_107, s_108, s_109, s_110, s_111, s_112, s_113, s_114, s_115, s_116, s_117, s_118, s_119, s_120,
                s_121, s_122, s_123, s_124, s_125, s_126, s_127, s_128, s_129, s_130,
            ]
        self.t_order_group_function = \
            [
                t_001, t_002, t_003, t_004, t_005, t_006, t_007, t_008, t_009, t_010, t_011, t_012, t_013, t_014, t_015,
                t_016, t_017, t_018, t_019, t_020, t_021, t_022, t_023, t_024, t_025, t_026, t_027, t_028, t_029, t_030,
                t_031, t_032, t_033, t_034, t_035, t_036, t_037, t_038, t_039, t_040, t_041, t_042, t_043, t_044, t_045,
                t_046, t_047, t_048, t_049, t_050, t_051, t_052, t_053, t_054, t_055, t_056, t_057, t_058, t_059, t_060,
                t_061, t_062, t_063, t_064, t_065, t_066, t_067, t_068, t_069, t_070, t_071, t_072, t_073, t_074,
            ]
        group_order_file_path = os.path.join('gp_3x_internal_data', 'group_order.xlsx')
        self.f_order_group_function_order = (
                pd.read_excel(group_order_file_path, sheet_name='f')['index'] - 1).tolist()  # 减1是为了基团序号和列表索引对上,因为python列表里的索引是从0开始的
        self.s_order_group_function_order = (
                pd.read_excel(group_order_file_path, sheet_name='s')['index'] - 1).tolist()  # 减1是为了基团序号和列表索引对上
        self.t_order_group_function_order = (
                pd.read_excel(group_order_file_path, sheet_name='t')['index'] - 1).tolist()  # 减1是为了基团序号和列表索引对上

    def count_a_mol(self, mol, clear_mode=False, add_note=False, add_smiles=False):
        init_smi = mol
        try:
            if isinstance(mol, str):
                mol = Chem.MolFromSmiles(mol)
            self.result = self.init_result.copy()
            if add_note:
                # self.result['note'] = ''
                self.result['note'] = Chem.MolToSmiles(mol)
            if add_smiles:
                self.result['smiles'] = Chem.MolToSmiles(mol)
            self.count_1st_order_groups(mol=mol, add_note=add_note)
            self.count_2nd_order_groups(mol=mol)
            self.count_3rd_order_groups(mol=mol)
            if clear_mode:
                # 清爽模式,不显示没有统计到的基团
                self.result = {k: v for k, v in self.result.items() if v}
            return self.result
        except:
            print(f'Error! There is something wrong when counting {init_smi}, please check it.')
            return self.init_result.copy()

    def count_mols(self, smiles_file_path, count_result_file_path='count_result.csv', add_note=False, add_smiles=False):  # todo mpi 并行？
        print('reading the input file...')
        if smiles_file_path.endswith('.txt'):
            smiles_iterator = list(open(smiles_file_path))
        elif smiles_file_path.endswith('.xlsx'):
            smiles_iterator = pd.read_excel(smiles_file_path)['smiles']
        elif smiles_file_path.endswith('.csv'):
            smiles_iterator = pd.read_csv(smiles_file_path)['smiles']
        else:
            raise NotImplementedError(
                'ERROR: The file type cannot be read, use the.txt/.xlsx/.csv file as the input file.')

        mol_number = len(smiles_iterator)
        print('Done, totally detected {} molecules, start counting...'.format(mol_number))
        count_result_dict_list = []
        for i in tqdm(smiles_iterator):
            count_result_dict_list.append(self.count_a_mol(i, add_note=add_note, add_smiles=add_smiles))
        print('Done!')
        print('writing to csv...')
        result = pd.DataFrame(count_result_dict_list)
        result.to_csv(count_result_file_path, index_label='index')
        print('Done!')
        return result

    def count_mols_mpi(self, smiles_file_path, count_result_file_path='count_result.csv', add_note=False, add_smiles=False, n_jobs=1, batch_size='auto'):
        print('reading the input file...')
        if smiles_file_path.endswith('.txt'):
            smiles_iterator = list(open(smiles_file_path))
        elif smiles_file_path.endswith('.xlsx'):
            smiles_iterator = pd.read_excel(smiles_file_path)['smiles']
        elif smiles_file_path.endswith('.csv'):
            smiles_iterator = pd.read_csv(smiles_file_path)['smiles']
        else:
            raise NotImplementedError(
                'ERROR: The file type cannot be read, use the.txt/.xlsx/.csv file as the input file.')

        mol_number = len(smiles_iterator)
        print('Done, totally detected {} molecules, start counting...'.format(mol_number))
        task = [delayed(self.count_a_mol)(i, add_note=add_note, add_smiles=add_smiles) for i in smiles_iterator]
        count_result_dict_list = Parallel(n_jobs=n_jobs, batch_size=batch_size)(task)
        print('Done!')
        print('writing to csv...')
        result = pd.DataFrame(count_result_dict_list)
        result.to_csv(count_result_file_path, index_label='index')
        print('Done!')
        return result

    def get_group_fingerprint(self, mol):
        count_result = self.count_a_mol(mol=mol, clear_mode=False, add_note=False)
        return [value for key, value in count_result.items()]

    def count_1st_order_groups(self, mol, add_note=False):
        atoms_index = set([i for i in range(len(mol.GetAtoms()))])  # 所有原子序号的集合
        used_atoms_index = set()
        for function_index in self.f_order_group_function_order:
            function = self.f_order_group_function[function_index]
            _, tuple_of_match_tuples = function(mol)
            for matched_index in tuple_of_match_tuples:
                if used_atoms_index.intersection(matched_index) == set():
                    used_atoms_index = used_atoms_index.union(matched_index)
                    self.result[function.__name__] += 1
            if used_atoms_index == atoms_index:
                # early stop 统计第一顺序基团时，当所有原子都被使用后提前停止
                break

        if used_atoms_index != atoms_index:
            warning = 'WARING: For {}, The first order groups do not cover the whole molecule! The results may not be reliable!'.format(
                Chem.MolToSmiles(mol))
            print(warning)
            if add_note:
                if self.result.get('note', 0):
                    self.result['note'] += '\t'
                    self.result['note'] += warning
                else:
                    self.result['note'] = warning
        return None

    def count_2nd_order_groups(self, mol):
        s_order_group_in_mol = []
        for function_index in self.s_order_group_function_order:
            function = self.s_order_group_function[function_index]
            _, tuple_of_match_tuples = function(mol)
            for matched_index in tuple_of_match_tuples:
                matched_index = set(matched_index)
                for used in s_order_group_in_mol:  # for else 语句
                    if matched_index.issubset(used):
                        break
                else:
                    s_order_group_in_mol.append(matched_index)
                    self.result[function.__name__] += 1
        return None

    def count_3rd_order_groups(self, mol):  # 跟count_2nd_order_groups几乎完全一样
        t_order_group_in_mol = []
        for function_index in self.t_order_group_function_order:
            function = self.t_order_group_function[function_index]
            _, tuple_of_match_tuples = function(mol)
            for matched_index in tuple_of_match_tuples:
                matched_index = set(matched_index)
                for used in t_order_group_in_mol:  # for else 语句
                    if matched_index.issubset(used):
                        break
                else:
                    t_order_group_in_mol.append(matched_index)
                    self.result[function.__name__] += 1
        return None


# if __name__ == '__main__':
#     print('debug gp_3x_counter.py ...')
#     import time
#
#     t1 = time.time()
#     # m = Chem.MolFromSmiles('Cc1ncc[nH]1')
#     c = Counter()
#     # result = c.count_a_mol(m, clear_mode=True)
#     # print(result)
#     #
#     # print(c.get_group_fingerprint(m))
#
#     c.count_mols_mpi(smiles_file_path=os.path.join('gp_3x_test_mol', 'SMILES.txt'), count_result_file_path='count_result.csv', add_note=True,
#                      n_jobs=4, batch_size='auto')
#
#
#     t2 = time.time()
#     print(t2 - t1)