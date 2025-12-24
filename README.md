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
 
## Dataset Interpretation & Labeling Strategy

The selected Kaggle dataset contains real-world rice samples with varying levels of impurities
(stones, insects, broken particles).

Since **almost all samples contain some level of impurity**, the dataset was reframed into a
binary quality classification problem:

- **Good**: Rice samples with minimal impurities, acceptable for consumption.
- **Bad**: Rice samples with heavy contamination, stones, or visible insects.

This reframing reflects a **practical quality inspection scenario**, where the goal is to
decide whether a rice batch passes or fails quality checks rather than detecting a perfectly clean sample.
