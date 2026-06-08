class NotMemoryError(Exception):
    pass


class ValidationError(NotMemoryError):
    pass


class RollbackError(NotMemoryError):
    pass


class ConflictError(NotMemoryError):
    pass


class StorageError(NotMemoryError):
    pass


class HashChainError(NotMemoryError):
    pass
