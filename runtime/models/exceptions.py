class ModelError(Exception):
    """Base model exception."""


class ModelNotFound(ModelError):
    """Raised when a model cannot be found."""


class InvalidModel(ModelError):
    """Raised when a model is invalid."""