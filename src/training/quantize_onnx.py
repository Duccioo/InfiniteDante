"""
ONNX Dynamic Quantization for InfiniteDante
===========================================
Quantizes model.onnx to INT8 weights (model_int8.onnx),
drastically reducing file size (~75% reduction) and speeding up inference
both in browser WASM and on CPU runtimes.
"""

import os
import sys
import time
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "model")
INPUT_ONNX = os.path.join(MODEL_DIR, "model.onnx")
OUTPUT_ONNX = os.path.join(MODEL_DIR, "model_int8.onnx")


def quantize():
    if not os.path.exists(INPUT_ONNX):
        print(f"Error: {INPUT_ONNX} not found.")
        sys.exit(1)

    orig_size_mb = os.path.getsize(INPUT_ONNX) / (1024 * 1024)
    print(f"Original ONNX model: {INPUT_ONNX} ({orig_size_mb:.2f} MB)")

    print("Running dynamic INT8 quantization...")
    start_t = time.time()
    quantize_dynamic(
        model_input=INPUT_ONNX,
        model_output=OUTPUT_ONNX,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
    )
    elapsed = time.time() - start_t

    quant_size_mb = os.path.getsize(OUTPUT_ONNX) / (1024 * 1024)
    ratio = (1 - quant_size_mb / orig_size_mb) * 100
    print(f"Quantization complete in {elapsed:.2f}s!")
    print(f"Quantized model: {OUTPUT_ONNX} ({quant_size_mb:.2f} MB)")
    print(f"Size reduction: -{ratio:.1f}%")

    # Verification: run sample inference on both models
    print("\n--- Verifying Output Consistency ---")
    sess_orig = ort.InferenceSession(INPUT_ONNX, providers=["CPUExecutionProvider"])
    sess_quant = ort.InferenceSession(OUTPUT_ONNX, providers=["CPUExecutionProvider"])

    # Create dummy token sequence of length 32
    dummy_input = np.random.randint(0, 512, size=(1, 32), dtype=np.int64)

    input_name = sess_orig.get_inputs()[0].name
    out_orig = sess_orig.run(None, {input_name: dummy_input})[0]
    out_quant = sess_quant.run(None, {input_name: dummy_input})[0]

    # Measure differences
    diff = np.abs(out_orig - out_quant)
    mean_diff = np.mean(diff)
    max_diff = np.max(diff)
    # Correlation between top predictions
    orig_top = np.argmax(out_orig[0, -1, :])
    quant_top = np.argmax(out_quant[0, -1, :])

    print(f"Mean absolute difference: {mean_diff:.4f}")
    print(f"Max absolute difference: {max_diff:.4f}")
    print(f"Original top predicted token at last pos: {orig_top}")
    print(f"Quantized top predicted token at last pos: {quant_top}")

    # Benchmark speed
    print("\n--- Speed Comparison (50 iterations) ---")
    t0 = time.time()
    for _ in range(50):
        _ = sess_orig.run(None, {input_name: dummy_input})
    t_orig = (time.time() - t0) / 50 * 1000

    t0 = time.time()
    for _ in range(50):
        _ = sess_quant.run(None, {input_name: dummy_input})
    t_quant = (time.time() - t0) / 50 * 1000

    print(f"Original FP32 latency: {t_orig:.2f} ms/step")
    print(f"Quantized INT8 latency: {t_quant:.2f} ms/step")
    print(f"Speedup: {t_orig / t_quant:.2f}x")

    return OUTPUT_ONNX


if __name__ == "__main__":
    quantize()
