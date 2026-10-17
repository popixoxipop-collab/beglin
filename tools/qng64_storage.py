"""Exact serialized qNg64 group-64 weight storage accounting (no dependencies).

One group contains 64 signed n-bit codes (n*8 bytes in bit planes) plus one
float32 scale (4 bytes). This is a serialized/on-disk payload estimate; it does
NOT include MLX repack, host/device duplicates, alignment, metadata, KV caches,
CUDA/Metal temporary buffers, nor measured runtime memory.
"""

GROUP = 64
MIN_N = 2
MAX_N = 15
# The actual BEGLIN qng64_dense.c API rejects n=4: that precision uses q4g64.
# n=8 remains its own qNg64 encoding on CPU, distinct from q8g64.
QNG64_CPU_BITS = frozenset(n for n in range(MIN_N, MAX_N + 1) if n != 4)
# Mirrored from vdsp-engine tools/backend_adapters.py, quant_format qng64_g64_ef_v1.
# GPU 4-bit uses q4g64; GPU 8-bit uses q8g64, NOT qNg64(n=4/8).
# Mirror native mlx_moe.cpp and tools/backend_adapters.py; no runtime claim.
QNG64_MLX_METAL_BITS = frozenset([2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15])


class QuantizationConfigurationError(ValueError):
    """An invalid qNg64 format/shape/backend is not silently approximated."""


def _integer(label, value):
    if type(value) is not int:
        raise QuantizationConfigurationError(f"{label} must be an integer")
    return value


def check_n(n):
    n = _integer("n", n)
    if not MIN_N <= n <= MAX_N or n == 4:
        raise QuantizationConfigurationError("BEGLIN qNg64 supports n=2..15 except n=4 (use q4g64)")
    return n


def check_backend_n(backend, n):
    n = check_n(n)
    if backend == "cpu":
        allowed = QNG64_CPU_BITS
    elif backend == "mlx_metal":
        allowed = QNG64_MLX_METAL_BITS
    else:
        raise QuantizationConfigurationError(f"unknown backend {backend!r}")
    if n not in allowed:
        raise QuantizationConfigurationError(
            f"qNg64(n={n}) does not have a declared {backend} backend path"
        )
    return n


def packed_weight_bytes(elements, n, group=GROUP):
    """Exact qNg64 code+scale bytes for aligned tensor, not MLX resident bytes."""
    n = check_n(n)
    elements = _integer("elements", elements)
    group = _integer("group", group)
    if group != GROUP:
        raise QuantizationConfigurationError("this qNg64 encoder requires group=64")
    if elements <= 0 or elements % group:
        raise QuantizationConfigurationError("positive tensor elements must be divisible by 64")
    # A group has n*64/8 = 8*n code bytes and 4 scale bytes.
    groups = elements // group
    return groups * (8 * n + 4)


def effective_bits_per_weight(n, group=GROUP):
    """qNg64 bit-plane n bits + one F32 scale PER 64 weights: n + 0.5 bpw."""
    check_n(n)
    if group != GROUP:
        raise QuantizationConfigurationError("this qNg64 encoder requires group=64")
    return n + 32.0 / group
