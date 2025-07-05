# Xyntra

## Project Description  
**Xyntra** is an automatic **kernel-fusion compiler pass** written in safe Rust.  
It ingests ONNX / TorchScript graphs, pattern-matches common op-chains, and emits **one fused GPU kernel** through `wgpu` (cross-platform WGSL) or optional CUDA PTX.  
The project explores graph rewriting, GPU occupancy modelling, and autotuned code-generation while keeping the entire pipeline **100 % `unsafe`-free**.

---

## Technologies & Dependencies

### **🦀 Core Technologies**
- **Rust 2024 Edition** – starting point; everything else will grow organically

### **📦 External Dependencies**
- _TBD_ – no external crates yet

---

## Features & Roadmap

### **🔧 Core Infrastructure & Foundations**
- [ ] Type-safe primitives – `NodeId`, `TensorShape`, `OpKind`, `Graph`
- [ ] Error-enum with recoverable vs fatal classes  
- [ ] Config loader – CLI flags & `fusion.toml`
- [ ] Modular crate layout – `xyntra-core`, `xyntra-cli`, `xyntra-ir`

### **📡 Graph Ingestion & Export**
- [ ] ONNX parser – load `.onnx` into internal IR  
- [ ] TorchScript loader – parse `.pt` via `tch-rs`
- [ ] IR serialisation – export to DOT / JSON for debugging  
- [ ] Fused-graph writer – emit reduced node graph snapshots

### **🧩 IR, Pattern Matching & Scheduling**
- [ ] `egg`-based e-graph integration – rewrite rules & saturation loop  
- [ ] Declarative fusion DSL – macro for `matmul -> gelu -> dropout`
- [ ] Scheduling heuristics – cost model for fusion candidates  
- [ ] Fusion legality checker – shape, dtype, broadcast guards

### **⚡ Kernel Code Generation**
- [ ] WGSL backend – emit compute shaders for `wgpu`  
- [ ] CUDA PTX backend – optional NV path behind `--backend ptx`
- [ ] Shared-memory tiling – configurable tile/block sizes  
- [ ] Vectorisation pass – `vec4<f32>` style loads/stores  
- [ ] Mixed-precision support (FP16/BF16) _(stretch)_

### **🚀 Autotuning & Performance**
- [ ] Parameter search harness – Bayesian optimiser over tile sizes  
- [ ] GPU occupancy analysis – register & SM utilisation metrics  
- [ ] Latency histogram – HDR log, p50/p95/p99 prints  
- [ ] Flamegraphs – CPU-side hotspots with `cargo flamegraph`
- [ ] Roofline model script – FLOP/s vs bandwidth chart _(stretch)_

### **🔒 Correctness & Validation**
- [ ] Golden unit tests – compare fused vs unfused outputs  
- [ ] Gradient checks – optional back-prop correctness suite  
- [ ] Edge-case library – broadcast, dynamic shapes, odd strides  
- [ ] Numerical tolerance config – FP32 / FP16 epsilon thresholds

### **📊 Observability & Diagnostics**
- [ ] Structured tracing spans – `tracing` crate with GPU timestamps  
- [ ] `--trace` CLI flag – dump kernel timeline to JSON  
- [ ] Occupancy dashboard – live CLI table of SM usage _(stretch)_

### **🛠️ Bench & Test Harness**
- [ ] Micro-bench harness – single op-chain latency  
- [ ] Model-zoo benchmarks – BERT, ResNet, ViT comparison  
- [ ] Determinism suite – random seeds & output hashes  
- [ ] CI matrix – MSRV check, clippy, fmt, criterion

### **🧰 Developer eXperience (DX)**
- [ ] `cargo xtask` or `justfile` – shortcuts (`just fuse resnet.onnx`)  
- [ ] Pre-commit hook – `cargo fmt && cargo clippy --fix`  
- [ ] `make dev` alias – spin-up CI-like environment locally

### **📦 Packaging & Release**
- [ ] GitHub Release action – build macOS, Linux, Windows binaries  
- [ ] Publish `xyntra-core` & `xyntra-cli` to *crates.io*  
- [ ] SemVer policy & `CHANGELOG.md` generation  
- [ ] Signed tags + GPG release checklist

### **🔗 Framework Plugins**
- [ ] PyTorch 2 `torch.xyntra.compile()` drop-in backend  
- [ ] ONNX Runtime execution-provider stub (`libxyntra_ep.so`)

### **📚 Docs & Examples**
- [ ] Quick-start guide – clone → build → fuse tiny MLP  
- [ ] Architecture diagram – ASCII / Mermaid / SVG  
- [ ] Fusion logs demo – before/after latency screenshot

### **🗃️ Model-Zoo Benchmarks**
- [ ] Scripted download + benchmark of BERT-Base, ResNet-50, ViT-Tiny, GPT-2  
- [ ] Auto-generated result table in README via CI

### **🌐 Stretch Goals & Research Paths**
- [ ] Horizontal fusion across attention blocks  
- [ ] Dynamic-shape specialisation & cache  
- [ ] Triton IR interoperability adapter  
- [ ] WebAssembly demo – run fused WGSL in browser  
- [ ] Meta-scheduler – ML-predicted tile sizes

### **🤝 DevOps & Community**
- [ ] GitHub Actions pipeline – lint, test, benches, release  
- [ ] Dual MIT / Apache-2 licence – broad adoption  
- [ ] `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue/PR templates  
- [ ] GitHub Discussions – Q&A, roadmap voting  
- [ ] Annotated blog series – graph rewriting, GPU tuning deep-dives

---
