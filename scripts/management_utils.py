from functools import wraps
import warnings


def Deperacated(func):
    @wraps(func)
    def inner(*args, **kwargs):
        func(*args, **kwargs)
        warnings.warn("Deperacated", UserWarning)
    return inner
