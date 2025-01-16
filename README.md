# Fine-Tune Llama 2 with LoRA and QLoRA

This repository provides a script for fine-tuning the Llama 2 model using Low-Rank Adaptation (LoRA) and Quantized LoRA (QLoRA) techniques. The fine-tuned model can be used for domain-specific tasks while significantly reducing computational costs and memory requirements.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)

---

## Overview

Fine-tuning large language models (LLMs) like Llama 2 can be resource-intensive. LoRA and QLoRA are Parameter-Efficient Fine-Tuning (PEFT) techniques that optimize this process by freezing most model parameters and only training a few additional ones. This approach allows for efficient fine-tuning even on systems with limited resources.

---

## Features

- Fine-tuning Llama 2 models with LoRA and QLoRA.
- Support for macOS MPS acceleration or CPU fallback.
- Configurable training hyperparameters via YAML files.
- Automatic saving of fine-tuned and merged models.
- Text generation pipeline for testing the fine-tuned model.

---

## Requirements

Ensure the following dependencies are installed:

- Python 3.8+
- PyTorch (with MPS support for macOS or CUDA for GPUs)
- Transformers
- Datasets
- TRL
- PEFT
- YAML

You can install all dependencies using:

```bash
pip install -r requirements.txt
