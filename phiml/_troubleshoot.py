from contextlib import contextmanager
from os.path import dirname

import packaging.version


def assert_minimal_config():  # raises AssertionError
    import sys
    assert sys.version_info.major == 3 and sys.version_info.minor >= 6, "Φ-ML requires Python 3.6 or newer to run"

    try:
        import numpy
    except ImportError:
        raise AssertionError("Φ-ML is unable to run because NumPy is not installed.")
    try:
        import scipy
    except ImportError:
        raise AssertionError("Φ-ML is unable to run because SciPy is not installed.")
    from . import math
    with math.NUMPY:
        a = math.ones()
        math.assert_close(a + a, 2)


def troubleshoot():
    from . import __version__
    return f"Φ-ML {__version__} at {dirname(__file__)}\n"\
           f"PyTorch: {troubleshoot_torch()}\n"\
           f"Jax: {troubleshoot_jax()}\n"\
           f"TensorFlow: {troubleshoot_tensorflow()}\n"  # TF last so avoid VRAM issues


def troubleshoot_tensorflow():
    from . import math
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # only errors
    try:
        import tensorflow
    except ImportError:
        return "Not installed."
    tf_version = f"{tensorflow.__version__} at {dirname(tensorflow.__file__)}"
    try:
        import tensorflow_probability
    except ImportError:
        return f"Installed ({tf_version}) but module 'tensorflow_probability' missing. Some functions may be unavailable, such as math.median() and math.quantile(). To install it, run  $ pip install tensorflow-probability"
    try:
        from . import tf
    except BaseException as err:
        return f"Installed ({tf_version}) but not available due to internal error: {err}"
    try:
        gpu_count = len(tf.TENSORFLOW.list_devices('GPU'))
    except BaseException as err:
        return f"Installed ({tf_version}) but device initialization failed with error: {err}"
    with tf.TENSORFLOW:
        try:
            math.assert_close(math.ones() + math.ones(), 2)
            # TODO cuDNN math.convolve(math.ones(batch=8, x=64), math.ones(x=4))
        except BaseException as err:
            return f"Installed ({tf_version}) but tests failed with error: {err}"
    if gpu_count == 0:
        return f"Installed ({tf_version}), {gpu_count} GPUs available."
    else:
        from .backend.tensorflow._tf_cuda_resample import librariesLoaded
        if librariesLoaded:
            cuda_str = 'CUDA kernels available.'
        else:
            import platform
            if platform.system().lower() != 'linux':
                cuda_str = f"Optional TensorFlow CUDA kernels not available and compilation not recommended on {platform.system()}. GPU will be used nevertheless."
            else:
                cuda_str = f"Optional TensorFlow CUDA kernels not available. GPU will be used nevertheless. Clone the Φ-ML source from GitHub and run 'python setup.py tf_cuda' to compile them. See https://tum-pbs.github.io/PhiML/Installation_Instructions.html"
        return f"Installed ({tf_version}), {gpu_count} GPUs available.\n{cuda_str}"


def troubleshoot_torch():
    from . import math
    try:
        import torch
    except ImportError:
        return "Not installed."
    torch_version = f"{torch.__version__} at {dirname(torch.__file__)}"
    try:
        from . import torch as torch_
    except BaseException as err:
        return f"Installed ({torch_version}) but not available due to internal error: {err}"
    try:
        gpu_count = len(torch_.TORCH.list_devices('GPU'))
    except BaseException as err:
        return f"Installed ({torch_version}) but device initialization failed with error: {err}"
    with torch_.TORCH:
        try:
            math.assert_close(math.ones() + math.ones(), 2)
        except BaseException as err:
            return f"Installed ({torch_version}) but tests failed with error: {err}"
    if torch_version.startswith('1.10.'):
        return f"Installed ({torch_version}), {gpu_count} GPUs available. This version has known bugs with JIT compilation. Recommended: 1.11+ or 1.8.2 LTS"
    if torch_version.startswith('1.9.'):
        return f"Installed ({torch_version}), {gpu_count} GPUs available. You may encounter problems importing torch.fft. Recommended: 1.11+ or 1.8.2 LTS"
    return f"Installed ({torch_version}), {gpu_count} GPUs available."


def troubleshoot_jax():
    from . import math
    try:
        import jax
        import jaxlib
    except ImportError:
        return "Not installed."
    version = f"jax {jax.__version__} at {dirname(jax.__file__)}, jaxlib {jaxlib.__version__}"
    try:
        from . import jax as jax_
    except BaseException as err:
        return f"Installed ({version}) but not available due to internal error: {err}"
    try:
        gpu_count = len(jax_.JAX.list_devices('GPU'))
    except BaseException as err:
        return f"Installed ({version}) but device initialization failed with error: {err}"
    with jax_.JAX:
        try:
            math.assert_close(math.ones() + math.ones(), 2)
        except BaseException as err:
            return f"Installed ({version}) but tests failed with error: {err}"
    if packaging.version.parse(jax.__version__) < packaging.version.parse('0.2.20'):
        return f"Installed ({version}), {gpu_count} GPUs available. This is an old version of Jax that may not support all required features, e.g. sparse matrices."
    return f"Installed ({version}), {gpu_count} GPUs available."


@contextmanager
def plot_solves():
    """
    While `plot_solves()` is active, certain performance optimizations and algorithm implementations may be disabled.
    """
    from . import math
    import pylab
    cycle = pylab.rcParams['axes.prop_cycle'].by_key()['color']
    with math.SolveTape(record_trajectories=True) as solves:
        try:
            yield solves
        finally:
            for i, result in enumerate(solves):
                assert isinstance(result, math.SolveInfo)
                from .math._tensors import disassemble_tree
                _, (residual,) = disassemble_tree(result.residual)
                residual_mse = math.mean(math.sqrt(math.sum(residual ** 2)), residual.shape.without('trajectory'))
                residual_mse_max = math.max(math.sqrt(math.sum(residual ** 2)), residual.shape.without('trajectory'))
                # residual_mean = math.mean(math.abs(residual), residual.shape.without('trajectory'))
                residual_max = math.max(math.abs(residual), residual.shape.without('trajectory'))
                pylab.plot(residual_mse.numpy(), label=f"{i}: {result.method}", color=cycle[i % len(cycle)])
                pylab.plot(residual_max.numpy(), '--', alpha=0.2, color=cycle[i % len(cycle)])
                pylab.plot(residual_mse_max.numpy(), alpha=0.2, color=cycle[i % len(cycle)])
                print(f"Solve {i}: {result.method} ({1000 * result.solve_time:.1f} ms)\n"
                      f"\t{result.solve}\n"
                      f"\t{result.msg}\n"
                      f"\tConverged: {result.converged.trajectory[-1]}\n"
                      f"\tDiverged: {result.diverged.trajectory[-1]}\n"
                      f"\tIterations: {result.iterations.trajectory[-1]}\n"
                      f"\tFunction evaulations: {result.function_evaluations.trajectory[-1]}")
            pylab.yscale('log')
            pylab.ylabel("Residual: MSE / max / individual max")
            pylab.xlabel("Iteration")
            pylab.title(f"Solve Convergence")
            pylab.legend(loc='upper right')
            pylab.savefig(f"pressure-solvers-FP32.png")
            pylab.show()


def count_tensors_in_memory(min_print_size: int = None):
    import sys
    import gc
    from .math import Tensor

    gc.collect()
    total = 0
    bytes = 0
    for obj in gc.get_objects():
        try:
            if isinstance(obj, Tensor):
                total += 1
                size = obj.shape.volume * obj.dtype.itemsize
                bytes += size
                if isinstance(min_print_size, int) and size >= min_print_size:
                    print(f"Tensor '{obj}' ({sys.getrefcount(obj)} references)")
                    # referrers = gc.get_referrers(obj)
                    # print([type(r) for r in referrers])
        except Exception:
            pass
    print(f"There are {total} Φ-ML Tensors with a total size of {bytes / 1024 / 1024:.1f} MB")
