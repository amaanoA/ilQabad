#!/usr/bin/env python3
"""Download and verify ONNX models for ilQabad face recognition system.

This script downloads the required models for:
- Face detection (YuNet)
- Face recognition (MobileFaceNet/ArcFace)
- Liveness detection (MiniFASNet)

Usage:
    poetry run python scripts/download_models.py

If automatic download fails, see models/README.md for manual download instructions.
"""

import subprocess
import sys
from pathlib import Path
from typing import TypedDict

import onnxruntime as ort


class ModelInfo(TypedDict):
    """Model information dictionary."""

    url: str
    path: Path
    description: str
    min_size: int  # Minimum expected file size in bytes
    license: str
    manual_instructions: str


# Base directory for models
MODELS_DIR = Path(__file__).parent.parent / "models"

# Model definitions with download URLs and metadata
MODELS: dict[str, ModelInfo] = {
    "yunet": {
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "path": MODELS_DIR / "detection" / "yunet.onnx",
        "description": "YuNet face detection with 5-point landmarks",
        "min_size": 200_000,  # ~233KB expected
        "license": "Apache-2.0",
        "manual_instructions": (
            "Download from OpenCV Zoo:\n"
            "https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/\n"
            "Click 'face_detection_yunet_2023mar.onnx' then 'Download raw file'"
        ),
    },
    "mobilefacenet": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-11.onnx",
        "path": MODELS_DIR / "recognition" / "mobilefacenet.onnx",
        "description": "ArcFace ResNet100 for 512-dim face embeddings",
        "min_size": 1_000_000,  # ~240MB expected for resnet100
        "license": "MIT",
        "manual_instructions": (
            "Download from ONNX Model Zoo:\n"
            "https://github.com/onnx/models/tree/main/validated/vision/body_analysis/arcface\n"
            "Or use Hugging Face: https://huggingface.co/models?search=arcface+onnx"
        ),
    },
    "minifas_2_7": {
        "url": "https://github.com/Kazuhito00/Silent-Face-Anti-Spoofing-ONNX/raw/main/model/2.7_80x80_MiniFASNetV2.onnx",
        "path": MODELS_DIR / "liveness" / "minifas_2_7.onnx",
        "description": "MiniFASNet v2 anti-spoofing (2.7x scale, 80x80)",
        "min_size": 300_000,  # ~400KB expected
        "license": "Apache-2.0",
        "manual_instructions": (
            "Download pre-converted ONNX from:\n"
            "https://github.com/Kazuhito00/Silent-Face-Anti-Spoofing-ONNX\n"
            "Or convert from PyTorch: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing"
        ),
    },
    "minifas_4_0": {
        "url": "https://github.com/Kazuhito00/Silent-Face-Anti-Spoofing-ONNX/raw/main/model/4_0x80x80_MiniFASNetV1SE.onnx",
        "path": MODELS_DIR / "liveness" / "minifas_4_0.onnx",
        "description": "MiniFASNet v1SE anti-spoofing (4.0x scale, 80x80)",
        "min_size": 300_000,  # ~400KB expected
        "license": "Apache-2.0",
        "manual_instructions": (
            "Download pre-converted ONNX from:\n"
            "https://github.com/Kazuhito00/Silent-Face-Anti-Spoofing-ONNX\n"
            "Or convert from PyTorch: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing"
        ),
    },
}

# Alternative URLs if primary fails
ALTERNATIVE_URLS: dict[str, list[str]] = {
    "yunet": [
        "https://huggingface.co/opencv/opencv_zoo/resolve/main/face_detection_yunet_2023mar.onnx",
    ],
    "mobilefacenet": [
        "https://huggingface.co/rocca/insightface-buffalo_l/resolve/main/w600k_r50.onnx",
    ],
    "minifas_2_7": [
        "https://github.com/hairymax/Face-AntiSpoofing/raw/main/saved_models/2.7_80x80_MiniFASNetV2.onnx",
    ],
    "minifas_4_0": [
        "https://github.com/hairymax/Face-AntiSpoofing/raw/main/saved_models/4_0x80x80_MiniFASNetV1SE.onnx",
    ],
}


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Human-readable size string (e.g., "1.5 MB").
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def download_with_curl(url: str, dest: Path, timeout: int = 120) -> bool:
    """Download file using curl command.

    Args:
        url: Source URL to download from.
        dest: Destination path to save file.
        timeout: Request timeout in seconds.

    Returns:
        True if download successful, False otherwise.
    """
    print(f"  Downloading from: {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            [
                "curl",
                "-L",  # Follow redirects
                "-f",  # Fail silently on HTTP errors
                "-s",  # Silent mode
                "-S",  # Show errors
                "--connect-timeout",
                "30",
                "--max-time",
                str(timeout),
                "-o",
                str(dest),
                url,
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 10,
        )

        if result.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
            print(f"  Downloaded: {format_size(dest.stat().st_size)}")
            return True
        else:
            if dest.exists():
                dest.unlink()  # Remove partial download
            print(f"  curl failed: {result.stderr.strip() or 'Unknown error'}")
            return False

    except subprocess.TimeoutExpired:
        print("  Download timed out")
        if dest.exists():
            dest.unlink()
        return False
    except FileNotFoundError:
        print("  curl not found, trying wget...")
        return download_with_wget(url, dest, timeout)
    except Exception as e:
        print(f"  Error: {e}")
        return False


def download_with_wget(url: str, dest: Path, timeout: int = 120) -> bool:
    """Download file using wget command as fallback.

    Args:
        url: Source URL to download from.
        dest: Destination path to save file.
        timeout: Request timeout in seconds.

    Returns:
        True if download successful, False otherwise.
    """
    try:
        result = subprocess.run(
            [
                "wget",
                "-q",  # Quiet
                "--timeout",
                str(timeout),
                "-O",
                str(dest),
                url,
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 10,
        )

        if result.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
            print(f"  Downloaded: {format_size(dest.stat().st_size)}")
            return True
        else:
            if dest.exists():
                dest.unlink()
            print(f"  wget failed: {result.stderr.strip() or 'Unknown error'}")
            return False

    except Exception as e:
        print(f"  wget error: {e}")
        return False


def download_with_fallback(
    name: str, model_info: ModelInfo, alternatives: list[str] | None = None
) -> bool:
    """Try downloading from primary URL, then alternatives.

    Args:
        name: Model name for logging.
        model_info: Model information dictionary.
        alternatives: List of alternative URLs to try.

    Returns:
        True if any download successful, False otherwise.
    """
    # Try primary URL
    if download_with_curl(model_info["url"], model_info["path"]):
        return True

    # Try alternatives
    if alternatives:
        for alt_url in alternatives:
            print("  Trying alternative URL...")
            if download_with_curl(alt_url, model_info["path"]):
                return True

    return False


def verify_onnx_model(path: Path) -> dict | None:
    """Load ONNX model and return input/output information.

    Args:
        path: Path to the ONNX model file.

    Returns:
        Dictionary with model info if successful, None otherwise.
    """
    if not path.exists():
        return None

    try:
        # Use CPU provider for verification
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

        inputs = []
        for inp in session.get_inputs():
            inputs.append({"name": inp.name, "shape": inp.shape, "type": inp.type})

        outputs = []
        for out in session.get_outputs():
            outputs.append({"name": out.name, "shape": out.shape, "type": out.type})

        return {
            "file_size": path.stat().st_size,
            "inputs": inputs,
            "outputs": outputs,
        }

    except Exception as e:
        print(f"  Verification error: {e}")
        return None


def print_model_info(name: str, info: dict) -> None:
    """Print model information in formatted way.

    Args:
        name: Model name.
        info: Model verification info dictionary.
    """
    print(f"  File size: {format_size(info['file_size'])}")
    print("  Inputs:")
    for inp in info["inputs"]:
        print(f"    - {inp['name']}: {inp['shape']} ({inp['type']})")
    print("  Outputs:")
    for out in info["outputs"]:
        print(f"    - {out['name']}: {out['shape']} ({out['type']})")


def print_manual_instructions(name: str, model_info: ModelInfo) -> None:
    """Print manual download instructions.

    Args:
        name: Model name.
        model_info: Model information dictionary.
    """
    print(f"\n  MANUAL DOWNLOAD REQUIRED for {name}:")
    print(f"  Target path: {model_info['path']}")
    print(f"  {model_info['manual_instructions']}")


def main() -> int:
    """Download and verify all models.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    print("=" * 60)
    print("ilQabad Model Downloader")
    print("=" * 60)
    print()

    # Ensure directories exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "detection").mkdir(exist_ok=True)
    (MODELS_DIR / "recognition").mkdir(exist_ok=True)
    (MODELS_DIR / "liveness").mkdir(exist_ok=True)

    results: dict[str, dict] = {}
    manual_needed: list[str] = []

    for name, model_info in MODELS.items():
        print(f"\n[{name.upper()}] {model_info['description']}")
        print("-" * 50)

        # Check if already downloaded and valid
        if model_info["path"].exists():
            size = model_info["path"].stat().st_size
            if size >= model_info["min_size"]:
                print(f"  Already exists: {format_size(size)}")
                info = verify_onnx_model(model_info["path"])
                if info:
                    results[name] = {"status": "exists", "info": info}
                    print("  Verification: PASSED")
                    print_model_info(name, info)
                    continue
                else:
                    print("  File exists but failed verification")

        # Download model
        alternatives = ALTERNATIVE_URLS.get(name)
        if download_with_fallback(name, model_info, alternatives):
            # Verify download
            size = model_info["path"].stat().st_size
            if size < model_info["min_size"]:
                print(
                    f"  WARNING: File smaller than expected "
                    f"({format_size(size)} < {format_size(model_info['min_size'])})"
                )

            info = verify_onnx_model(model_info["path"])
            if info:
                results[name] = {"status": "downloaded", "info": info}
                print("  Verification: PASSED")
                print_model_info(name, info)
            else:
                results[name] = {"status": "failed_verification", "info": None}
                print("  Verification: FAILED")
                manual_needed.append(name)
        else:
            results[name] = {"status": "failed_download", "info": None}
            print("  Download: FAILED (network may be restricted)")
            manual_needed.append(name)

    # Print summary
    print("\n")
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()
    print(f"{'Model':<20} {'Status':<15} {'Size':<12} {'Inputs':<20}")
    print("-" * 67)

    all_success = True
    for name, result in results.items():
        status = result["status"]
        if status in ("exists", "downloaded"):
            info = result["info"]
            size = format_size(info["file_size"])
            inputs = ", ".join(i["name"] for i in info["inputs"])
            status_str = "OK" if status == "exists" else "Downloaded"
        else:
            size = "N/A"
            inputs = "N/A"
            status_str = "FAILED"
            all_success = False

        print(f"{name:<20} {status_str:<15} {size:<12} {inputs:<20}")

    # Print manual download instructions if needed
    if manual_needed:
        print("\n")
        print("=" * 60)
        print("MANUAL DOWNLOAD INSTRUCTIONS")
        print("=" * 60)
        for name in manual_needed:
            print_manual_instructions(name, MODELS[name])

        print("\n\nAfter downloading, place files in the paths shown above.")
        print("Then re-run this script to verify the models.")

    print()

    if all_success:
        print("All models downloaded and verified successfully!")
        return 0
    else:
        print("\nSome models require manual download.")
        print("See models/README.md for detailed instructions.")
        return 1


def verify_only() -> int:
    """Verify existing models without downloading.

    Returns:
        Exit code (0 if all present, 1 otherwise).
    """
    print("Verifying existing models...")
    print()

    all_present = True
    for name, model_info in MODELS.items():
        path = model_info["path"]
        if path.exists():
            info = verify_onnx_model(path)
            if info:
                print(f"[OK] {name}: {format_size(info['file_size'])}")
            else:
                print(f"[INVALID] {name}: exists but failed to load")
                all_present = False
        else:
            print(f"[MISSING] {name}: {path}")
            all_present = False

    return 0 if all_present else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        raise SystemExit(verify_only())
    raise SystemExit(main())
