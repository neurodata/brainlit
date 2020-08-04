import contextlib
from pathlib import Path
from joblib.parallel import BatchCompletionCallBack


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar given as argument
    """

    class TqdmBatchCompletionCallback(BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def check_type(input, types):
    if not isinstance(input, types):
        raise TypeError((f"{input} should be {types}, not {type(input)}."))


def check_iterable_type(input, types):
    if not all(isinstance(i, types) for i in input):
        raise TypeError((f"{input} elements should be {types}."))


def check_size(input, allow_float=True):
    check_type(input, (list, tuple))
    if len(input) != 3:
        raise ValueError(f"{input} must have x, y, z dimensions")
    if allow_float:
        check_iterable_type(input, (int, float))
    else:
        check_iterable_type(input, int)


def check_precomputed(input):
    check_type(input, str)
    prefix = input.split(":")[0]
    if prefix not in ["file", "s3", "gc"]:
        raise NotImplementedError("only file, s3, and gc prefixes supported")
    if prefix == "file":
        if not Path(input.split(":")[1][2:]).is_dir():
            raise ValueError(f"{input} is not a directory.")
