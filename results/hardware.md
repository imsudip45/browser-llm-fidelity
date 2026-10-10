# Test Environment & Hardware Specifications

The benchmarks in this directory (`results/`) were conducted on the following laptop hardware:

| Component | Specification |
| :--- | :--- |
| **Device Type** | Laptop (Windows PC) |
| **Operating System** | Windows 11 (64-bit, Build x86_64) |
| **Processor (CPU)** | 13th Gen Intel(R) Core(TM) i5-13420H (12 vCPUs, up to 4.60 GHz) |
| **Discrete GPU (dGPU)** | NVIDIA GeForce RTX 2050 Laptop GPU (4 GB GDDR6 VRAM, Ampere architecture) |
| **Integrated GPU (iGPU)** | Intel(R) UHD Graphics (Gen-12LP architecture, shared system memory) |
| **System Memory (RAM)** | 8 GB |
| **Browser Tested** | Google Chrome 139.0.0.0 (Windows NT 10.0; Win64; x64) |
| **Host Runtime Baseline** | Node.js v20+ (`native-node-cpu/fp32`) |

---

### GPU Assignment Mapping
- **`nvidia-chrome` runs**: Google Chrome assigned to **High performance** (NVIDIA GeForce RTX 2050).
- **`intel-chrome` runs**: Google Chrome assigned to **Power saving** (Intel UHD Graphics).
- **`cpu-chrome-check` run**: Transformers.js executed via browser WebAssembly (WASM) on Intel Core i5-13420H CPU.
