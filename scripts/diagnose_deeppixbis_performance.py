"""Diagnose DeePixBiS performance bottlenecks."""

import os
import platform
import time
from pathlib import Path

import numpy as np

# Check if model exists
MODEL_PATH = Path("models/liveness/deeppixbis.onnx")
if not MODEL_PATH.exists():
    print(f"Model not found at {MODEL_PATH}")
    exit(1)

print("=" * 60)
print("DeePixBiS Performance Diagnostics")
print("=" * 60)

# System info
print("\n[SYSTEM INFO]")
print(f"  Platform: {platform.platform()}")
print(f"  Processor: {platform.processor() or 'Unknown'}")
print(f"  Python: {platform.python_version()}")

# Check CPU count
cpu_count = os.cpu_count()
print(f"  CPU cores: {cpu_count}")

# Check ONNX Runtime
import onnxruntime as ort

print(f"\n[ONNX Runtime]: {ort.__version__}")
print(f"  Available providers: {ort.get_available_providers()}")

# Load model and time it
print("\n[TIMING BREAKDOWN]")

# 1. Model loading time
start = time.perf_counter()
session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
load_time = time.perf_counter() - start
print(f"  Model loading: {load_time*1000:.1f}ms")

# Get input/output info
input_name = session.get_inputs()[0].name
print(f"  Input name: {input_name}")
print(f"  Input shape: {session.get_inputs()[0].shape}")

# 2. Create test image
start = time.perf_counter()
test_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
create_time = time.perf_counter() - start
print(f"  Create test image: {create_time*1000:.2f}ms")

# 3. Preprocessing time (simulate what the implementation does)
start = time.perf_counter()
# Resize (already 224x224)
img_float = test_image.astype(np.float32) / 255.0
# ImageNet normalization
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
normalized = (img_float - mean) / std
# Transpose HWC to CHW
transposed = np.transpose(normalized, (2, 0, 1))
# Add batch dimension
batched = np.expand_dims(transposed, axis=0).astype(np.float32)
preprocess_time = time.perf_counter() - start
print(f"  Preprocessing: {preprocess_time*1000:.2f}ms")

# 4. First inference (cold)
start = time.perf_counter()
outputs = session.run(None, {input_name: batched})
first_inference_time = time.perf_counter() - start
print(f"  First inference (cold): {first_inference_time*1000:.1f}ms")

# 5. Warm-up inferences
print("\n  Warm-up runs:")
for i in range(3):
    start = time.perf_counter()
    outputs = session.run(None, {input_name: batched})
    warm_time = time.perf_counter() - start
    print(f"    Run {i+1}: {warm_time*1000:.1f}ms")

# 6. Measure consistent inference (10 runs)
print("\n  Benchmark (10 runs after warmup):")
times = []
for i in range(10):
    start = time.perf_counter()
    outputs = session.run(None, {input_name: batched})
    elapsed = time.perf_counter() - start
    times.append(elapsed * 1000)

print(f"    Min: {min(times):.1f}ms")
print(f"    Max: {max(times):.1f}ms")
print(f"    Avg: {sum(times)/len(times):.1f}ms")
print(f"    Std: {np.std(times):.1f}ms")

# 7. Output info
print("\n[OUTPUT INFO]")
for i, out in enumerate(outputs):
    print(f"  Output {i}: shape={out.shape}, min={out.min():.4f}, max={out.max():.4f}")


# 8. Full pipeline test (preprocess + inference + postprocess)
print("\n[FULL PIPELINE] (preprocess + inference + postprocess):")


def full_pipeline(image):
    # Preprocess
    img_float = image.astype(np.float32) / 255.0
    normalized = (img_float - mean) / std
    transposed = np.transpose(normalized, (2, 0, 1))
    batched = np.expand_dims(transposed, axis=0).astype(np.float32)

    # Inference
    outputs = session.run(None, {input_name: batched})

    # Postprocess
    binary_score = float(outputs[1][0, 0])
    pixel_map = outputs[0][0, 0]  # Squeeze to 14x14

    return binary_score, pixel_map


# Warmup
for _ in range(3):
    full_pipeline(test_image)

# Measure
pipeline_times = []
for _ in range(10):
    start = time.perf_counter()
    score, pmap = full_pipeline(test_image)
    pipeline_times.append((time.perf_counter() - start) * 1000)

print(f"  Min: {min(pipeline_times):.1f}ms")
print(f"  Max: {max(pipeline_times):.1f}ms")
print(f"  Avg: {sum(pipeline_times)/len(pipeline_times):.1f}ms")

# 9. Summary
print("\n" + "=" * 60)
print("[SUMMARY]")
print("=" * 60)
avg_inference = sum(times) / len(times)
if avg_inference < 50:
    print(f"  Inference is FAST ({avg_inference:.0f}ms) - likely GPU/good CPU")
elif avg_inference < 200:
    print(f"  Inference is MODERATE ({avg_inference:.0f}ms) - typical for CPU")
else:
    print(f"  Inference is SLOW ({avg_inference:.0f}ms) - check environment")

if first_inference_time * 1000 > avg_inference * 3:
    print(
        f"  First inference is {first_inference_time*1000/avg_inference:.1f}x slower (JIT warmup)"
    )
    print("     -> Test should include warmup runs before timing")

print("\n[RECOMMENDATIONS]:")
if avg_inference > 100:
    print(f"  - Adjust test threshold to {int(avg_inference * 2)}ms for this environment")
    print("  - First inference is always slower due to JIT compilation")
    print("  - Consider adding warmup runs in the test")
