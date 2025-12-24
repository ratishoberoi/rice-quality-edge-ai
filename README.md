# Rice Grain Quality Detection - Edge AI

This project implements an end-to-end Edge AI computer vision pipeline
to detect rice grain quality (Whole, Broken, Impurity)
directly on-device without any cloud dependency.

## Objective
Build a lightweight, optimized computer vision model for mobile edge deployment.

## Constraints
- No cloud inference
- Edge optimized model under 5MB
- Real-time on-device inference

## Project Structure
data/        - raw and processed datasets
training/    - model training and evaluation
edge/        - model conversion and quantization
inference/   - standalone inference script
mobile_app/  - mobile demo using TFLite
models/      - saved baseline and edge models

## Status
Phase 0 completed - project structure and environment ready.
