from functools import wraps
import warnings


def Deperacated(func):
    @wraps(func)
    def inner(*args, **kwargs):
        warnings.warn("Deperacated", UserWarning)
        return func(*args, **kwargs)
    return inner
