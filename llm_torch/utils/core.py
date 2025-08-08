import inspect
from typing import Any, Type, Optional, Dict


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def make_get_function(namespace: Dict[str, Any]):
    """Returns a 'get' function bound to a specific namespace (e.g., globals())."""
    def get(identifier: Optional[Any], default_class: Optional[Type] = None) -> Type:
        """
        Retrieve a class based on an identifier, using the provided namespace.

        Parameters:
            identifier: None, a class type, or a string representing a class name.
            default_class: The class to return if identifier is None.

        Returns:
            The corresponding class type.

        Raises:
            ValueError: If the string identifier does not match any known class.
            TypeError: If the identifier is not None, a class, or a string.
        """
        if identifier is None:
            if default_class:
                return default_class
            raise ValueError("Identifier is None and no default class provided.")

        if inspect.isclass(identifier):
            return identifier

        if isinstance(identifier, str):
            cls = namespace.get(identifier)
            if cls is None or not inspect.isclass(cls):
                raise ValueError(f"No class found for identifier '{identifier}'.")
            return cls

        raise TypeError(f"Expected None, a class, or a string, but got {type(identifier).__name__}.")

    return get
