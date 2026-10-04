#!/bin/sh
cd /home/user/workspace/bfid
M=onnx-community/Qwen2.5-0.5B-Instruct
for d in fp32 q4 fp16; do node run_native.mjs $M $d all > logs_$d.txt 2>&1; done
