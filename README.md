# Efficient Attention Vision Transformers for Monocular Depth Estimation

[![Paper](https://img.shields.io/badge/Paper-View-blue)](https://doi.org/10.21203/rs.3.rs-6328112/v1)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

This repository provides the implementation and experimental framework for the paper:

> **Efficient Attention Vision Transformers for Monocular Depth Estimation on Resource-Limited Hardware**
> Claudio Schiavella, Lorenzo Cirillo, Lorenzo Papa, Paolo Russo, Irene Amerini
> [DOI: 10.21203/rs.3.rs-6328112/v1](https://doi.org/10.21203/rs.3.rs-6328112/v1)

---

## 🔎 Overview

This project investigates the integration of **efficient attention mechanisms** into Vision Transformers (ViTs) for **Monocular Depth Estimation (MDE)**.
The main goal is to balance **prediction quality** and **inference speed** on resource-limited hardware through structural optimizations of attention modules.

---

## 📆 Requirements

* [Docker](https://www.docker.com/) or [Podman](https://podman.io/)
* GPU with NVIDIA Container Toolkit
* Download datasets manually:

  * [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
  * [KITTI Raw Depth](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction)

---

## 🐳 Docker Setup

Build and launch the environment by running:

```bash
bash run_docker
```

Follow the on-screen prompts to configure model, dataset path, and training options.
This will start training inside the container using your local data and project code.

---

## 🧲 Evaluation

Testing and evaluation are handled via:

```bash
automation.py
```

You can use this script within the container to automate evaluation across models and configurations.

---

## 📉 Entropy Analysis (Encoder/Decoder)

To run entropy-based analysis of attention modules, use the following `podman` command:

```bash
podman run \
  -v WORK_DIR:/work/project \
  -v DATA_DIR:/work/data \
  --device nvidia.com/gpu=all \
  --ipc host \
  docker.io/claudioschi21/thesis_alcor_cuda11.8:latest \
  /usr/bin/python3 work/project/enc_dec_analysis.py
```

Replace `WORK_DIR` with the path to your local project and `DATA_DIR` with the path to your dataset.

---

## 📁 Repository Structure

* `automation.py` – Evaluation entry point
* `enc_dec_analysis.py` – Attention entropy analysis
* `models/` – Vision Transformer architectures (METER, PixelFormer, NeWCRFs)
* `scripts/` – Training helpers and utilities
* `docker/` – Environment configuration

---

## 📜 License

This work is licensed under the **Creative Commons Attribution 4.0 International License**.
You are free to share and adapt the material for any purpose, even commercially, with proper attribution.

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

---

## 📖 Citation

If you use this code or data, please cite the following paper:

```bibtex
@article{schiavella2025efficient,
  title={Efficient Attention Vision Transformers for Monocular Depth Estimation on Resource-Limited Hardware},
  author={Schiavella, Claudio and Cirillo, Lorenzo and Papa, Lorenzo and Russo, Paolo and Amerini, Irene},
  journal={Research Square},
  year={2025},
  doi={10.21203/rs.3.rs-6328112/v1}
}
```

---