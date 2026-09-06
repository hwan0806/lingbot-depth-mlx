"""Verify preconverted FP32/BF16 checkpoints; no weight conversion."""
from pathlib import Path

from validate import sha256

CONFIG_SHA = "d4cd52c33e0d64fb3589ccc675c474c4b6f2e3ade7302b5d7e7842ca956bc309"
PRECISION_SHA = "602b8638ac0dca5dc74b80c3fb6dcd3216fdbf862fea60faee235926d1943616"
WEIGHT_SHA = {
    "fp32": "cd87dc1f9bb379edffdef98d38595b2ec6640f90524a8c3e820fd09dec805123",
    "bf16": "2b67ce4896586a4c59e4bc578754439ad22b457998fc493c247bb594d2e02134",
}
MODEL_IDS = {
    'fp32': 'lingbot-depth-mlx-fp32',
    'bf16': 'lingbot-depth-mlx-mixed-bf16-fp32',
}


def verify_checkpoint(directory: Path) -> str:
    directory = Path(directory)
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("missing or symlinked MLX checkpoint directory")
    if (directory / "quantization.json").exists():
        raise ValueError("Use the released FP32 or BF16 checkpoint, not quantized weights")
    precision = "bf16" if (directory / "precision.json").exists() else "fp32"
    expected = {"config.json": CONFIG_SHA, "weights.safetensors": WEIGHT_SHA[precision]}
    if precision == "bf16":
        expected["precision.json"] = PRECISION_SHA
    for name, digest in expected.items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or sha256(path) != digest:
            raise ValueError("MLX checkpoint checksum mismatch: " + name)
    return precision
