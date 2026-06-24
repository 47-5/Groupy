"""Small chemistry helpers shared by Groupy modules."""

from rdkit import Chem, rdBase

from groupy.exceptions import InvalidSmilesError


def ensure_mol(mol):
    """Return an RDKit molecule from a SMILES string or molecule-like object."""
    if isinstance(mol, str):
        with rdBase.BlockLogs():
            parsed = Chem.MolFromSmiles(mol)
        if parsed is None:
            raise InvalidSmilesError(f"Invalid SMILES: {mol}")
        return parsed
    if mol is None or not hasattr(mol, "GetAtoms"):
        raise TypeError("Expected a SMILES string or an RDKit molecule.")
    return mol
