"""Custom exceptions used by Groupy library APIs."""


class GroupyError(Exception):
    """Base class for expected Groupy runtime errors."""


class InvalidSmilesError(GroupyError, ValueError):
    """Raised when a SMILES string cannot be parsed into a molecule."""


class ConversionError(GroupyError):
    """Raised when a molecule or file conversion step fails."""
