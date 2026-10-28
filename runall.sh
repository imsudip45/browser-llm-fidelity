#!/usr/bin/env bash
set -e
node run_native.mjs onnx-community/Qwen2.5-0.5B-Instruct fp32 all
node run_native.mjs onnx-community/Qwen2.5-0.5B-Instruct fp16 all
node run_native.mjs onnx-community/Qwen2.5-0.5B-Instruct q4 all
node run_native.mjs onnx-community/Qwen2.5-0.5B-Instruct q4f16 all
