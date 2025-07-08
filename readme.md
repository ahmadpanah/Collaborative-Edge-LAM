# Collaborative Edge LAM Framework: A Python Simulation

This repository contains a Python simulation of the framework presented in the research paper: **"Collaborative Deployment of Large AI Models on the Edge: A Microservice Approach to Heterogeneous Training and Quantized Inference"**.

This project provides a functional, high-level implementation of the core architectural components and algorithms, demonstrating how to manage the lifecycle of Large AI Models (LAMs) in resource-constrained and heterogeneous edge computing environments.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

---

## Table of Contents

- [Collaborative Edge LAM Framework: A Python Simulation](#collaborative-edge-lam-framework-a-python-simulation)
  - [Table of Contents](#table-of-contents)
  - [Problem Statement](#problem-statement)
  - [Key Features](#key-features)
  - [System Architecture](#system-architecture)
  - [Project Structure](#project-structure)
  - [Setup and Installation](#setup-and-installation)
  - [How to Run the Simulation](#how-to-run-the-simulation)
  - [Core Concepts Implemented](#core-concepts-implemented)
    - [1. Heterogeneity-Aware Federated Training (FedFT-H)](#1-heterogeneity-aware-federated-training-fedft-h)
    - [2. Precision-Aware Microservice Inference](#2-precision-aware-microservice-inference)
    - [3. Efficient Federated Unlearning](#3-efficient-federated-unlearning)
  - [Simulation Output](#simulation-output)
  - [Contributing](#contributing)

## Problem Statement

Deploying Large AI Models (LAMs) like GPT-4 and Gemini on real-time Internet-of-Things (IoT) devices is a significant challenge. This is due to the severe mismatch between the massive computational and memory requirements of LAMs and the limited resources of edge devices. The problem is compounded by **deep system heterogeneity**, where devices vary widely in computational power (e.g., high-end GPUs vs. low-power MCUs) and supported numerical precisions (e.g., FP32 vs. INT8).

This framework provides a unified, modular, and adaptive solution to overcome these barriers.

## Key Features

This simulation implements the two core innovations presented in the paper:

-   **⚙️ Heterogeneity-Aware Training Framework**:
    -   Integrates **Parameter-Efficient Fine-Tuning (PEFT)** using Low-Rank Adaptation (LoRA) to drastically reduce communication overhead.
    -   Supports **Quantization-Aware Training (QAT)**, allowing devices with different hardware precisions (e.g., FP16, INT8) to collaborate on fine-tuning a single global model.
    -   Introduces a novel server-side aggregation mechanism that handles updates of varying ranks and precisions by de-quantizing and padding updates before averaging.

-   **⚡ Precision-Aware Inference Architecture**:
    -   Virtualizes LAM functionalities (like Mixture-of-Experts layers or Chain-of-Thought steps) into a portfolio of **multi-precision microservices**.
    -   Features a **dynamic orchestrator** that selects the optimal microservice variant (e.g., high-accuracy FP16 vs. low-latency INT8) for each task based on real-time Quality-of-Service (QoS) requirements.
    -   Uses **Lyapunov optimization** to balance the trade-offs between immediate performance (accuracy, latency) and long-term system stability (device load).

-   **🗑️ Efficient Federated Unlearning**:
    -   Implements a projection-based unlearning mechanism to efficiently remove a client's contribution from the global model without the prohibitive cost of full retraining, ensuring privacy and regulatory compliance.

## System Architecture

The framework is coordinated by a central Edge Server that manages two synergistic workflows: Collaborative Training and Dynamic Inference.

```
+-------------------------------------------------------------------------+
|                          EDGE SERVER (Orchestrator)                     |
|                                                                         |
|  +---------------------------+               +------------------------+ |
|  |   Heterogeneity-Aware     |   Global LoRA   |   Precision-Aware      | |
|  |      Aggregation          |<--------------->|    Inference Orchestrator| |
|  | (Algorithm 1)             |     Update      | (Algorithm 2)          | |
|  | - De-Quantization         |               | - Lyapunov Optimization| |
|  | - Adaptive Rank Aggregation |               | - QoS-based Scheduling | |
|  +---------------------------+               +------------------------+ |
|        ^             |                                    |              |
| Uplink |             | Downlink                           | Deployment   |
| (LoRA) |             | (LoRA)                             | Results      |
+--------|-------------|------------------------------------|--------------+
         |             |                                    |
         v             v                                    v
+-------------------------------------------------------------------------+
|                      HETEROGENEOUS EDGE NETWORK                         |
|                                                                         |
|  [COLLABORATIVE TRAINING]                     [DYNAMIC INFERENCE]       |
|                                                                         |
| +------------+  +------------+             +-------------+  +-----------+ |
| | Device 1   |  | Device 2   |             | Device 1    |  | Device 3  | |
| | (H-Tier)   |  | (M-Tier)   |             | (H-Tier)    |  | (L-Tier)  | |
| | r=16, FP16 |  | r=8, FP16  |             | m1_FP16     |  | m2_INT8   | |
| +------------+  +------------+             +-------------+  +-----------+ |
|                                                                         |
| +------------+  +------------+             +-------------+  |           | |
| | Device 3   |  | Device 4   |             | Device 2    |  |           | |
| | (L-Tier)   |  | (M-Tier)   |             | (M-Tier)    |  |           | |
| | r=4, INT8  |  |            |             | m3_FP16     |  |           | |
| +------------+  +------------+             +-------------+  +-----------+ |
+-------------------------------------------------------------------------+
```

## Project Structure

The codebase is organized into a modular structure to separate concerns.

```
collaborative_edge_lam/
├── main.py                     # Main script to run all simulations
├── config.py                   # Central configuration for devices, model, network
└── framework/
    ├── __init__.py
    ├── server.py               # Implements the EdgeServer orchestrator logic
    ├── device.py               # Implements the EdgeDevice client logic
    ├── microservice.py         # Defines microservice variants and the portfolio
    ├── lora.py                 # Helper data class for LoRA updates
    └── utils.py                # Logging and other utility functions
```

## Setup and Installation

This project uses only standard Python libraries (NumPy) and requires no special installation.

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/ahmadpanah/Collaborative-Edge-LAM.git
    cd Collaborative-Edge-LAM
    ```

2.  **Ensure you have Python 3.8+ installed.**

3.  (Optional but recommended) Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

## How to Run the Simulation

To run all three experiments described in the paper, simply execute the `main.py` script:

```sh
python collaborative_edge_lam/main.py
```

The script will print a detailed log of the simulation process and a summary of the results for each experiment, comparing our framework against the baselines.

## Core Concepts Implemented

### 1. Heterogeneity-Aware Federated Training (FedFT-H)

This workflow, detailed in `Algorithm 1` of the paper, allows devices of varying capabilities to train a model together.

-   **Client-Side (`device.py`)**:
    1.  A device receives the global LoRA model and **truncates** it to its maximum supported rank (`r`).
    2.  It simulates local training.
    3.  The resulting update is **quantized** to the device's native hardware precision (e.g., INT8).

-   **Server-Side (`server.py`)**:
    1.  The server receives heterogeneous LoRA updates (different ranks and precisions).
    2.  Each update is **de-quantized** to a common high-precision format (FP32).
    3.  Each update is **padded with zeros** to match the maximum rank (`r_max`) in the federation.
    4.  A **weighted federated average** is performed to create the new global model.

### 2. Precision-Aware Microservice Inference

This workflow, detailed in `Algorithm 2`, enables efficient, parallelized inference.

-   **Microservice Portfolio (`microservice.py`)**: The system maintains a portfolio of services, where each logical function (e.g., `CoT_Step`) exists in multiple versions (`FP16`, `INT8`, `Pruned_INT8`), each with different accuracy/performance trade-offs.

-   **Orchestration (`server.py`)**:
    1.  An incoming request with QoS constraints (e.g., `min_accuracy > 95%`) is received.
    2.  The orchestrator filters the portfolio to find all `(microservice, device)` pairs that satisfy the QoS.
    3.  It uses a **Lyapunov drift-plus-penalty** function to score each valid pair, balancing the immediate `PerformanceCost` (latency, energy) against the `QueueCost` (current device load).
    4.  The pair with the lowest objective score is selected, ensuring both good performance for the current task and long-term stability for the network.

### 3. Efficient Federated Unlearning

-   **Orthogonal Projection (`server.py`)**: Instead of retraining the entire model from scratch, we simulate the "right to be forgotten." The `federated_unlearning` method retrieves the forgotten client's last weighted contribution and effectively "subtracts" it from the global model. This process is orders of magnitude faster than retraining, as demonstrated in the simulation.

## Simulation Output

Running the `main.py` script will produce an output similar to the following, summarizing the results of the three experiments.

```
--- Training Performance Summary (after 50 rounds as per paper) ---
Training Strategy           Final Accuracy (%)        Total Comm. Cost (GB)
--------------------------------------------------------------------------------
Full-Model FedAvg           85.5%                     ~280.0 GB
Naive Fed-LoRA (r=4)        72.1%                     ~5.1 GB
Ours (FedFT-H)              84.9% (Simulated)         ~5.3 GB
FedFT-H achieves near full-model accuracy with >98% communication savings.

--- Inference Performance Summary (4-Step CoT Task) ---
Deployment Strategy         Avg. Latency (ms)         Memory Footprint
--------------------------------------------------------------------------------
Cloud-Centric               850.4                     N/A (Cloud-Side)
Monolithic Edge             400.0                     14.5 GB (on one device)
Ours (Microservice)         172.0                     4.2 GB (Total Active)
Our microservice approach shows a ~57.0% latency reduction through parallelization.

--- Federated Unlearning Efficacy & Cost Comparison ---
State / Method              Model Accuracy (%)        Time Cost
--------------------------------------------------------------------------------
Original Trained Model      84.9%                     N/A
Full Retraining             84.5%                     ~5 Hours
Ours (Orthogonal Unlearn)   84.3% (Simulated)         ~0.01 Seconds
Our unlearning method is orders of magnitude faster than retraining with minimal accuracy loss.
```


## Contributing

Contributions are welcome! If you have ideas for improvements, please open an issue to discuss what you would like to change. Pull requests are also appreciated.

---
This project is licensed under the MIT License.