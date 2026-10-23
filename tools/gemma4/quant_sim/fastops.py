"""Fast paths for the quantization simulator: the HIP Hessian kernel
(kernels.hip, built on first use with the ROCm SDK of the running PyTorch)
and GPTQ blocks replayed as HIP graphs."""

import ctypes
import hashlib
import os
import subprocess
import sysconfig
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
_lib = None


def _sdk():
    site = Path(sysconfig.get_paths()["purelib"])
    return site / "_rocm_sdk_devel"


def library():
    """Builds kernels.hip into a cached shared library and loads it."""
    global _lib
    if _lib is not None:
        return _lib
    src = HERE / "kernels.hip"
    digest = hashlib.sha1(src.read_bytes()).hexdigest()[:12]
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    out = cache / "gufo-quant-sim" / f"kernels-{digest}.so"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        sdk = _sdk()
        bitcode = sdk / "lib/llvm/amdgcn/bitcode"
        env = dict(os.environ, ROCM_PATH=str(sdk), HIP_PATH=str(sdk))
        cmd = [str(sdk / "bin/hipcc"), "-O3", "--offload-arch=gfx1151",
               "-std=c++17", "-shared", "-fPIC", f"--rocm-path={sdk}",
               f"--rocm-device-lib-path={bitcode}", str(src), "-o", str(out)]
        subprocess.run(cmd, check=True, env=env)
    _lib = ctypes.CDLL(str(out))
    _lib.SyrkAccumulate.restype = ctypes.c_int
    _lib.SyrkAccumulate.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_void_p, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_void_p]
    return _lib


def hessian_accumulate(h, x):
    """h (FP32 [K, K], upper triangle valid) += x^T x for x [T, K] FP32.
    Binary16 inputs scaled per column to max |x| = 1, FP32 products sums."""
    t, k = x.shape
    pad = (-t) % 32
    col = x.abs().amax(0).clamp_min(1e-30).float().contiguous()
    xs = (x / col).half()
    if pad:
        xs = torch.cat((xs, xs.new_zeros(pad, k)))
    xs = xs.t().contiguous()  # [K][T]: the kernel stages 16-byte token runs
    stream = torch.cuda.current_stream().cuda_stream
    rc = library().SyrkAccumulate(xs.data_ptr(), col.data_ptr(), h.data_ptr(),
                                  t + pad, k, stream)
    if rc != 0:
        raise RuntimeError(f"SyrkAccumulate failed ({rc}) for k={k}")


def symmetrize(h):
    """Copies the upper triangle into the lower one."""
    upper = torch.triu(h)
    return upper + upper.t() - torch.diag(torch.diag(h))
