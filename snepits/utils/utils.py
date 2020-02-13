from time import time


def timer(func):
    """ Timer decorator """

    def wrapper(*args, **kwargs):
        t0 = time()
        out = func(*args, **kwargs)
        t1 = time()
        return (t1 - t0), out

    return wrapper
