"""Quantization variants for sim.py.

A variant name selects a preset, optionally restricted to some projection
classes (the others stay BF16) with `@`:

    bf16, fp32              reference projections
    gguf:<name>             a GGUF's dequantized projections (--gguf name=path)
    w4rtn / w4gptq          FWHT-rotated symmetric int4, per-128 binary16 scale
    w4a4rtn / w4a4gptq      the same with rotated int4 activations per 128
    w4a8gptq                int8 activations per 128 (diagnostic)
    w8rtn                   rotated symmetric int8, per-128 scale
    w4rtn-norot             symmetric int4 without the rotation
    <preset>-g64            weight scale groups of 64 instead of 128
                            (more flags in Variant: -rN -agN -sm -cc)
    asym4rtn-norot          asymmetric int4 (scale and min) without rotation

    exl3k4 / exl3k3         EXL3 trellis at 4 / 3 bits (Kyojin's exllamav3)
    w4agptq / w4a4agptq     asymmetric int4 (binary16 scale and min) GPTQ
    hyb:<a>+<b>             prompt rows through <a>, reply rows through <b>
    hyb:<a>+<b>#ends3_glob  ... keeping <b> for the prompt in listed layers
    w4a4gptq@down           only the FFN down projection quantized
    w4a4gptq@qkv+gate_up    two classes quantized
    w4a4gptq/a=o+down       W4 everywhere, A4 only on o and down inputs

Rotation: x' = H(D x) in blocks of `ROT` values along K (D random signs per
layer and input), W' rows the same, so W x = W' x'.
"""

import math
import zlib

import torch

CLASSES = ("qkv", "o", "gate_up", "down")
LINEARS = {"qkv": ("q", "k", "v"), "o": ("o",), "gate_up": ("gate", "up"),
           "down": ("down",)}
GGUF_NAME = {"q": "attn_q", "k": "attn_k", "v": "attn_v", "o": "attn_output",
             "gate": "ffn_gate", "up": "ffn_up", "down": "ffn_down"}
ROT = 128
GROUP = 128
WEIGHT_CACHE = {}  # this layer's quantized weights by quantizer


def fwht(x, block=ROT):
    """Orthonormal Walsh-Hadamard transform over blocks of the last dim."""
    shape = x.shape
    y = x.reshape(-1, shape[-1] // block, block)
    h = 1
    while h < block:
        y = y.view(y.shape[0], y.shape[1], block // (2 * h), 2, h)
        a, b = y[:, :, :, 0, :], y[:, :, :, 1, :]
        y = torch.stack((a + b, a - b), dim=3)
        h *= 2
    return (y.reshape(shape) * (1.0 / math.sqrt(block)))


def signs(layer, cls, k, device):
    g = torch.Generator(device="cpu")
    g.manual_seed(zlib.crc32(f"{layer}:{cls}".encode()))
    s = torch.randint(0, 2, (k,), generator=g).float() * 2 - 1
    return s.to(device)


CLIP = torch.linspace(0.70, 1.0, 13)


def sym_quant(w, bits, group=GROUP, clip=True, scale_dtype=torch.float16):
    """Symmetric RTN per `group` along the last dim; MSE-searched clip."""
    qmax = 2 ** (bits - 1) - 1
    shape = w.shape
    g = w.reshape(-1, group)
    amax = g.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    ratios = CLIP if clip else torch.ones(1)
    best = None
    best_err = None
    for r in ratios.tolist():
        s = (amax * r / qmax).to(scale_dtype).float().clamp_min(1e-12)
        q = torch.clamp(torch.round(g / s), -qmax - 1, qmax) * s
        err = (q - g).pow(2).sum(-1, keepdim=True)
        if best is None:
            best, best_err = q, err
        else:
            take = err < best_err
            best = torch.where(take, q, best)
            best_err = torch.minimum(err, best_err)
    return best.reshape(shape)


def sym_scales(w, bits, scale_dtype=torch.float16):
    """Per-row MSE-searched scales for one group column block w [N, G]."""
    qmax = 2 ** (bits - 1) - 1
    amax = w.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    best_s = None
    best_err = None
    for r in CLIP.tolist():
        s = (amax * r / qmax).to(scale_dtype).float().clamp_min(1e-12)
        q = torch.clamp(torch.round(w / s), -qmax - 1, qmax) * s
        err = (q - w).pow(2).sum(-1, keepdim=True)
        if best_s is None:
            best_s, best_err = s, err
        else:
            take = err < best_err
            best_s = torch.where(take, s, best_s)
            best_err = torch.minimum(err, best_err)
    return best_s


def asym_quant(w, bits, group=GROUP):
    """Asymmetric RTN, BF16 scale and min (Strixite's q4g128)."""
    qmax = 2 ** bits - 1
    shape = w.shape
    g = w.reshape(-1, group)
    lo = g.amin(-1, keepdim=True)
    hi = g.amax(-1, keepdim=True)
    s = ((hi - lo) / qmax).bfloat16().float().clamp_min(1e-12)
    m = lo.bfloat16().float()
    q = torch.clamp(torch.round((g - m) / s), 0, qmax) * s + m
    return q.reshape(shape)


def tri_inv(lower, leaf=1024):
    """Inverse of a lower-triangular matrix, recursively in halves so most
    of the work is matrix products (2.4x torch's solve_triangular here)."""
    n = lower.shape[0]
    if n <= leaf:
        eye = torch.eye(n, device=lower.device, dtype=lower.dtype)
        return torch.linalg.solve_triangular(lower, eye, upper=False)
    h = (n // 2 + 127) // 128 * 128
    a = tri_inv(lower[:h, :h], leaf)
    c = tri_inv(lower[h:, h:], leaf)
    out = torch.zeros_like(lower)
    out[:h, :h] = a
    out[h:, h:] = c
    out[h:, :h] = -(c @ (lower[h:, :h] @ a))
    return out


def chol(m, leaf=2048):
    """Lower Cholesky factor, recursively in halves (raises LinAlgError
    like torch.linalg.cholesky when m is not positive definite)."""
    n = m.shape[0]
    if n <= leaf:
        return torch.linalg.cholesky(m)
    h = (n // 2 + 127) // 128 * 128
    l11 = chol(m[:h, :h], leaf)
    l21 = m[h:, :h] @ tri_inv(l11).t()
    l22 = chol(m[h:, h:] - l21 @ l21.t(), leaf)
    out = torch.zeros_like(m)
    out[:h, :h] = l11
    out[h:, :h] = l21
    out[h:, h:] = l22
    return out


def upper_inverse_factor(m):
    """U upper with m^-1 = U^T U: with J the order reversal, J m J = C C^T
    gives m = V V^T for upper V = J C J, so U = V^-1 = J C^-1 J. One
    Cholesky and one triangular inverse instead of two Choleskys and an
    explicit inverse."""
    rev = torch.arange(m.shape[0] - 1, -1, -1, device=m.device)
    c = chol(m[rev][:, rev].contiguous())
    return tri_inv(c)[rev][:, rev].contiguous()


def inverse_hessian_factor(h, damp):
    """Upper Cholesky factor of (H + d I)^-1, d = damp * mean(diag H).

    The 21504-wide down-projection Hessians can lose positive definiteness
    in FP32; retry in FP64, then with 10x and 100x damping."""
    k = h.shape[0]
    mean = torch.mean(torch.diag(h))
    eye = torch.eye(k, device=h.device)
    for scale in (1, 10, 100):
        for dtype in (torch.float32, torch.float64):
            try:
                hd = (h + eye * (damp * scale * mean)).to(dtype)
                try:
                    f = upper_inverse_factor(hd)
                except torch.linalg.LinAlgError:
                    raise
                except RuntimeError as e:
                    # hipSOLVER workspace allocation can fail with many
                    # variants resident; redo the factorization on the CPU.
                    if "ALLOC_FAILED" not in str(e):
                        raise
                    torch.cuda.empty_cache()
                    f = upper_inverse_factor(hd.cpu()).to(h.device)
                    print(f"gptq: k={k} factorized on the CPU", flush=True)
            except torch.linalg.LinAlgError:
                continue
            if scale != 1 or dtype != torch.float32:
                print(f"gptq: k={k} needed {dtype} and {scale}x damping",
                      flush=True)
            return f.float()
    raise RuntimeError(f"gptq: Hessian of width {k} not invertible")


ASYM_SHRINK = torch.linspace(0.80, 1.0, 9)


def asym_params(w, bits):
    """Per-row MSE-searched scale and min (binary16) for w [N, G]:
    w ~ q * s + m, q in [0, 2^bits - 1]."""
    qmax = 2 ** bits - 1
    lo = w.amin(-1, keepdim=True)
    hi = w.amax(-1, keepdim=True)
    mid = (lo + hi) / 2
    best = None
    best_err = None
    for r in ASYM_SHRINK.tolist():
        l = mid + (lo - mid) * r
        u = mid + (hi - mid) * r
        sc = ((u - l) / qmax).half().float().clamp_min(1e-12)
        m = l.half().float()
        q = torch.clamp(torch.round((w - m) / sc), 0, qmax) * sc + m
        err = (q - w).pow(2).sum(-1, keepdim=True)
        if best is None:
            best, best_err = (sc, m), err
        else:
            take = err < best_err
            best = (torch.where(take, sc, best[0]), torch.where(take, m, best[1]))
            best_err = torch.minimum(err, best_err)
    return best


def gptq_block(w1, hinv1, q1, err1, bits, group, asym, s1=None):
    """One GPTQ block in place: w1 [N, B] working weights (updated), hinv1
    [B, B] upper factor; writes dequantized q1 and scaled errors err1, and
    the symmetric scales to s1 [N, B / group] when given.
    Scales come from the weights as updated so far, at each group start."""
    qmax = 2 ** (bits - 1) - 1
    scale = low = None
    for i in range(w1.shape[1]):
        if i % group == 0:
            if asym:
                scale, low = asym_params(w1[:, i:i + group], bits)
            else:
                scale = sym_scales(w1[:, i:i + group], bits)
                if s1 is not None:
                    s1[:, i // group:i // group + 1] = scale
        x = w1[:, i:i + 1]
        if asym:
            q = torch.clamp(torch.round((x - low) / scale), 0,
                            2 ** bits - 1) * scale + low
        else:
            q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax) * scale
        q1[:, i:i + 1] = q
        e = (x - q) / hinv1[i, i]
        w1[:, i:] -= e @ hinv1[i:i + 1, i:]
        err1[:, i:i + 1] = e


class GptqGraph:
    """gptq_block captured once per shape and replayed: the block is ~1000
    small launches, which dominate when dispatched from Python."""

    def __init__(self, n, block, bits, group, asym, device):
        self.w1 = torch.zeros(n, block, device=device)
        self.hinv1 = torch.zeros(block, block, device=device)
        self.q1 = torch.zeros(n, block, device=device)
        self.err1 = torch.zeros(n, block, device=device)
        self.s1 = torch.zeros(n, block // group, device=device)
        args = (bits, group, asym, self.s1)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            gptq_block(self.w1.clone(), self.hinv1, self.q1, self.err1, *args)
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            gptq_block(self.w1, self.hinv1, self.q1, self.err1, *args)

    def run(self, w1, hinv1):
        self.w1.copy_(w1)
        self.hinv1.copy_(hinv1)
        self.graph.replay()
        return self.q1, self.err1, self.s1


GPTQ_GRAPHS = {}


def gptq(w, h, bits, group=GROUP, damp=0.01, block=GROUP, asym=False,
         graphs=True, scales=False):
    """GPTQ with static per-group scales (symmetric, or scale and min),
    columns in order. Blocks replay a captured graph when K divides into
    them (bit-identical to graphs=False). With scales=True (symmetric, K a
    multiple of the group) also returns the scales [N, K / group]."""
    w = w.float().clone()
    n, k = w.shape
    h = h.clone()
    dead = torch.diag(h) == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    hinv = inverse_hessian_factor(h, damp)
    out = torch.zeros_like(w)
    sc = torch.zeros(n, k // group, device=w.device) if scales else None
    runner = None
    if graphs and k % block == 0 and block % group == 0:
        key = (n, block, bits, group, asym, w.device)
        if key not in GPTQ_GRAPHS:
            GPTQ_GRAPHS[key] = GptqGraph(n, block, bits, group, asym, w.device)
        runner = GPTQ_GRAPHS[key]
    for b0 in range(0, k, block):
        b1 = min(b0 + block, k)
        hinv1 = hinv[b0:b1, b0:b1]
        if runner is not None:
            q1, err1, s1 = runner.run(w[:, b0:b1], hinv1)
        else:
            w1 = w[:, b0:b1].clone()
            q1 = torch.zeros_like(w1)
            err1 = torch.zeros_like(w1)
            s1 = torch.zeros(n, (b1 - b0 + group - 1) // group,
                             device=w.device)
            gptq_block(w1, hinv1, q1, err1, bits, group, asym, s1)
        out[:, b0:b1] = q1
        if sc is not None:
            sc[:, b0 // group:b1 // group] = s1
        w[:, b1:] -= err1 @ hinv[b0:b1, b1:]
    return (out, sc) if scales else out


def exl3_quant(w, h, bits, seed):
    """EXL3 round trip of w [N, K] given the input Hessian h [K, K]."""
    import sys
    sys.path.insert(0, "/home/franz/projects/kyojin")
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
    h_data = {"H": h.clone(), "count": 1, "finalized": False,
              "device": w.device}
    args = {"K": bits, "seed": seed, "devices": [w.device.index or 0],
            "mul1": True, "apply_out_scales": None}
    q = quantize_exl3(w.t().contiguous(), h_data, args, True)[0]
    return q.t().contiguous()


def act_quant(x, bits, group=GROUP):
    """Per-token symmetric activation quantization per `group`, f32 scale,
    MSE-searched clip (four candidates)."""
    qmax = 2 ** (bits - 1) - 1
    shape = x.shape
    g = x.reshape(-1, group)
    amax = g.abs().amax(-1, keepdim=True).clamp_min(1e-20)
    best = None
    best_err = None
    for r in (1.0, 0.9, 0.8, 0.7):
        s = amax * r / qmax
        q = torch.clamp(torch.round(g / s), -qmax - 1, qmax) * s
        err = (q - g).pow(2).sum(-1, keepdim=True)
        if best is None:
            best, best_err = q, err
        else:
            take = err < best_err
            best = torch.where(take, q, best)
            best_err = torch.minimum(err, best_err)
    return best.reshape(shape)


def half_matmul(x, w16):
    y = torch.mm(x.half(), w16.t()).float()
    if not torch.isfinite(y).all():
        y = torch.mm(x, w16.float().t())
    return y


class Prepared:
    """One projection class's evaluation weights."""

    def __init__(self, weights, bits_per_weight, rotate=None, act=None,
                 fp32=False):
        self.weights = weights  # name -> device tensor (rotated basis)
        self.bpw = bits_per_weight
        self.rotate = rotate  # x -> x' or None
        self.act = act  # x' -> fake-quantized x' or None
        self.fp32 = fp32
        self._cached = None

    def apply(self, name, x, layer):
        if self.rotate is not None:
            # q/k/v and gate/up share their input: rotate and quantize once.
            if self._cached is None or self._cached[0] is not x:
                xr = self.rotate(x)
                if self.act is not None:
                    xr = self.act(xr)
                self._cached = (x, xr)
            x = self._cached[1]
        elif self.act is not None:
            x = self.act(x)
        w = self.weights[name]
        if self.fp32:
            return torch.mm(x, w.t())
        return half_matmul(x, w)

    def bits(self):
        count = sum(w.numel() for w in self.weights.values())
        return self.bpw * count, count


FLAGS = ("g", "r", "ag", "sm", "cc")


def rotation_block(k, wanted):
    """Largest power-of-two block <= wanted that divides k."""
    block = 1
    while block * 2 <= wanted and k % (block * 2) == 0:
        block *= 2
    return block


class Variant:
    """<preset>[-g64][-r512][-ag64][-sm][-cc][@classes][/a=classes].

    -gN   weight scale groups of N (default 128)
    -rN   rotation blocks of up to N values (power of two dividing K)
    -agN  activation scale groups of N (default 128)
    -sm   SmoothQuant-style per-channel scales (alpha 0.5) before rotation
    -cc   calibrate on the chat-format set instead of the standard mix
    -c6   calibrate on bartowski's calibration-v6 (calib_file.py)"""

    def __init__(self, name, ggufs):
        self.name = name
        base, _, classes = name.partition("@")
        base, _, act_only = base.partition("/a=")
        self.classes = set(classes.split("+")) if classes else set(CLASSES)
        self.act_classes = (set(act_only.split("+")) if act_only
                            else set(CLASSES))
        assert self.classes <= set(CLASSES), name
        self.group = GROUP
        self.rot = ROT
        self.agroup = GROUP
        self.smooth = False
        self.calib_set = "std"
        parts = base.split("-")
        keep = [parts[0]]
        for part in parts[1:]:
            if part == "sm":
                self.smooth = True
            elif part == "cc":
                self.calib_set = "chat"
            elif part == "c6":
                self.calib_set = "v6"
            elif part.startswith("ag") and part[2:].isdigit():
                self.agroup = int(part[2:])
            elif part.startswith("g") and part[1:].isdigit():
                self.group = int(part[1:])
            elif part.startswith("r") and part[1:].isdigit():
                self.rot = int(part[1:])
            else:
                keep.append(part)
        base = "-".join(keep)
        self.base = base
        self.gguf = None
        self.rotated = False
        self.method = None
        self.wbits = None
        self.abits = None
        if base in ("bf16", "fp32"):
            pass
        elif base.startswith("exl3k"):
            # EXL3 (QTIP-style trellis, mul1 codebook, Hessian LDLQ), through
            # Kyojin's exllamav3 build.
            self.method = "exl3"
            self.wbits = int(base[5:])
        elif base.startswith("gguf:"):
            import sys
            sys.path.insert(0, "/home/franz/projects/strix-llama.cpp/gguf-py")
            import gguf
            self.gguf_mod = gguf
            self.gguf = gguf.GGUFReader(ggufs[base[5:]])
            self.gguf_tensors = {t.name: t for t in self.gguf.tensors}
        else:
            presets = {
                "w4rtn": (True, "rtn", 4, None),
                "w4gptq": (True, "gptq", 4, None),
                "w4a4rtn": (True, "rtn", 4, 4),
                "w4a4gptq": (True, "gptq", 4, 4),
                "w4a8gptq": (True, "gptq", 4, 8),
                "w4agptq": (True, "gptqa", 4, None),
                "w4a4agptq": (True, "gptqa", 4, 4),
                "w8rtn": (True, "rtn", 8, None),
                "w4rtn-norot": (False, "rtn", 4, None),
                "asym4rtn-norot": (False, "asym", 4, None),
            }
            self.rotated, self.method, self.wbits, self.abits = presets[base]
        self.calibrated = self.method in ("gptq", "gptqa", "exl3")
        self.needs_stats = self.smooth

    def describe(self):
        return {"base": self.base, "classes": sorted(self.classes),
                "act_classes": sorted(self.act_classes),
                "rotation": self.rot if self.rotated else None,
                "group": self.group, "act_group": self.agroup,
                "smooth": self.smooth, "calibration": self.calib_set,
                "method": self.method, "wbits": self.wbits,
                "abits": self.abits}

    def prepare(self, layer_index, cls, layer, hessians):
        """`hessians` maps calibration set -> class -> mean x x^T (raw
        inputs), or None for uncalibrated variants."""
        names = [n for n in LINEARS[cls] if n in layer.w]
        if self.base == "fp32":
            return Prepared({n: layer.w[n].float() for n in names}, 16,
                            fp32=True)
        if self.base == "bf16" or cls not in self.classes:
            return Prepared({n: layer.w16[n] for n in names}, 16)
        if self.gguf is not None:
            key = (layer_index, cls, self.base)
            if key not in WEIGHT_CACHE:
                if WEIGHT_CACHE and next(iter(WEIGHT_CACHE))[0] != layer_index:
                    WEIGHT_CACHE.clear()
                weights = {}
                bits = 0
                count = 0
                for n in names:
                    t = self.gguf_tensors[
                        f"blk.{layer_index}.{GGUF_NAME[n]}.weight"]
                    deq = self.gguf_mod.quants.dequantize(t.data, t.tensor_type)
                    w = torch.from_numpy(deq.copy()).to(layer.w[n].device)
                    assert w.shape == layer.w[n].shape, (n, w.shape)
                    weights[n] = w.half()
                    bits += t.n_bytes * 8
                    count += w.numel()
                WEIGHT_CACHE[key] = (weights, bits / count)
            weights, bpw = WEIGHT_CACHE[key]
            return Prepared(weights, bpw)
        dev = layer.w[names[0]].device
        k = layer.w[names[0]].shape[1]
        hess = None
        if self.calibrated or self.smooth:
            hess = hessians[self.calib_set][cls]
        # Input transform x' = R(D x / s) and weights W' rows R(D W s):
        # D random signs, s SmoothQuant scales (or 1), R block FWHT.
        col = torch.ones(k, device=dev)
        if self.smooth:
            x_rms = torch.diag(hess).clamp_min(1e-12).sqrt()
            w_max = torch.stack([layer.w[n].float().abs().amax(0)
                                 for n in names]).amax(0).clamp_min(1e-12)
            col = (x_rms.sqrt() / w_max.sqrt()).clamp(1e-4, 1e4)
        rotate = None
        wrot = None
        block = rotation_block(k, self.rot)
        if self.rotated:
            d = signs(layer_index, cls, k, dev)
            vx = d / col
            vw = d * col

            def rotate(x, vx=vx):
                return fwht(x * vx, block)

            def wrot(w, vw=vw):
                return fwht(w * vw, block)
        if hess is not None and self.rotated:
            # H' = R diag(vx) H diag(vx) R^T.
            hs = hess * vx[:, None] * vx[None, :]
            hess = fwht(fwht(hs, block).t().contiguous(), block).t().contiguous()
        # Variants that differ only in activations share their weights.
        key = (layer_index, cls, self.rotated, self.method, self.wbits,
               self.group, block, self.smooth, self.calib_set)
        if key not in WEIGHT_CACHE:
            if WEIGHT_CACHE and next(iter(WEIGHT_CACHE))[0] != layer_index:
                WEIGHT_CACHE.clear()
            weights = {}
            if self.method in ("gptq", "gptqa"):
                # GPTQ treats output rows independently given H, so the
                # projections sharing this input (q|k|v, gate|up) run as one
                # stacked matrix: identical results, one column sweep.
                w = torch.cat([layer.w[n].float() for n in names])
                if wrot is not None:
                    w = wrot(w)
                q = gptq(w, hess, self.wbits, group=self.group,
                         asym=self.method == "gptqa")
                row = 0
                for n in names:
                    rows = layer.w[n].shape[0]
                    weights[n] = q[row:row + rows].half()
                    row += rows
                names_left = []
            else:
                names_left = names
            for n in names_left:
                w = layer.w[n].float()
                if wrot is not None:
                    w = wrot(w)
                if self.method == "exl3":
                    q = exl3_quant(w, hess, self.wbits, layer_index)
                elif self.method == "rtn":
                    q = sym_quant(w, self.wbits, group=self.group)
                elif self.method == "asym":
                    q = asym_quant(w, self.wbits, group=self.group)
                elif self.method == "gptqa":
                    q = gptq(w, hess, self.wbits, group=self.group, asym=True)
                else:
                    q = gptq(w, hess, self.wbits, group=self.group)
                weights[n] = q.half()
            WEIGHT_CACHE[key] = weights
        weights = WEIGHT_CACHE[key]
        act = None
        if self.abits is not None and cls in self.act_classes:
            bits_a, group_a = self.abits, self.agroup

            def act(x, bits_a=bits_a, group_a=group_a):
                return act_quant(x, bits_a, group=group_a)
        if self.method == "exl3":
            n_out, k_in = layer.w[names[0]].shape
            bpw = self.wbits + 16 / n_out + 16 / k_in  # suh and svh vectors
        else:
            scale_bits = 32 if self.method in ("asym", "gptqa") else 16
            bpw = self.wbits + scale_bits / self.group
        return Prepared(weights, bpw, rotate=rotate, act=act)


SPLIT = None  # rows before this index are prompt rows (hybrid variants)


class HybridPrepared:
    """Prompt rows through one format, the reply rows through another."""

    def __init__(self, prefill, decode):
        self.prefill = prefill
        self.decode = decode

    def apply(self, name, x, layer):
        if SPLIT is None or SPLIT >= x.shape[0]:
            return self.prefill.apply(name, x, layer)
        if SPLIT <= 0:
            return self.decode.apply(name, x, layer)
        return torch.cat((self.prefill.apply(name, x[:SPLIT], layer),
                          self.decode.apply(name, x[SPLIT:], layer)))

    def bits(self):
        return self.prefill.bits()


class Hybrid:
    """hyb:<prefill>+<decode>, e.g. hyb:w4a4gptq+gguf:q4kxl: prompt tokens
    through the first variant, the scored reply through the second."""

    # Layers whose prompt rows keep the decode format too (#<spec>).
    KEEP = {
        "ends3": {0, 1, 2, 57, 58, 59},
        "glob": {5, 11, 17, 23, 29, 35, 41, 47, 53, 59},
    }

    def __init__(self, name, ggufs):
        self.name = name
        name, _, keep = name.partition("#")
        self.keep = set()
        for part in filter(None, keep.split("_")):
            self.keep |= self.KEEP[part]
        a, b = name[4:].split("+", 1)
        self.prefill = Variant(a, ggufs)
        self.decode = Variant(b, ggufs)
        self.calibrated = self.prefill.calibrated or self.decode.calibrated
        self.calib_sets = {v.calib_set for v in (self.prefill, self.decode)
                           if v.calibrated or v.smooth}

    def describe(self):
        return {"prefill": self.prefill.describe(),
                "decode": self.decode.describe()}

    def prepare(self, layer_index, cls, layer, hessians):
        decode = self.decode.prepare(layer_index, cls, layer, hessians)
        if layer_index in self.keep:
            return HybridPrepared(decode, decode)
        return HybridPrepared(
            self.prefill.prepare(layer_index, cls, layer, hessians), decode)


def calib_sets(variant):
    if isinstance(variant, Hybrid):
        return variant.calib_sets
    if variant.calibrated or variant.smooth:
        return {variant.calib_set}
    return set()


def make_variant(name, ggufs):
    if name.startswith("hyb:"):
        return Hybrid(name, ggufs)
    return Variant(name, ggufs)
