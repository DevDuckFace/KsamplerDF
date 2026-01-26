# KSampler DF (with 2 samplers and Refiner)

A custom ComfyUI node that splits sampling steps between a main sampler and a refiner sampler, with independent denoise control for each phase.

## Features

- **Split Steps**: Divide total steps between main sampler and refiner sampler
- **Dual Sampler Support**: Use different samplers/schedulers for main and refiner phases
- **Independent Denoise Control**: Separate denoise parameters for main and refiner phases
- **Refiner Intensity**: Control how much the refiner affects the final result (0.0 to 1.0)

## Installation

### Method 1: Git Clone (Recommended)

1. Open a terminal/command prompt
2. Navigate to your ComfyUI custom nodes folder:
   ```bash
   cd ComfyUI/custom_nodes
   ```
3. Clone this repository:
   ```bash
   git clone https://github.com/DevDuckFace/KsamplerDF.git
   ```
4. Restart ComfyUI

### Method 2: Manual Installation

1. Download this repository as a ZIP file
2. Extract the contents to `ComfyUI/custom_nodes/KsamplerDF/`
3. Restart ComfyUI

## Usage

Find the node in the **Add Node** menu under: `sampling` → `KSampler DF (with Refiner)`

### Parameters

| Parameter | Description |
|-----------|-------------|
| **model** | The model used for denoising |
| **positive** | Positive conditioning (prompt) |
| **negative** | Negative conditioning (negative prompt) |
| **latent_image** | The latent image to denoise |
| **seed** | Random seed for noise generation |
| **steps** | Total number of steps (split between main and refiner) |
| **cfg** | Classifier-Free Guidance scale |
| **sampler_name** | Sampler algorithm for the main phase |
| **scheduler** | Scheduler for the main phase |
| **denoise** | Denoise amount for the main phase (0.0 to 1.0) |
| **refiner_at_step** | Step at which to switch from main to refiner |
| **sampler_refiner** | Sampler algorithm for the refiner phase |
| **scheduler_refiner** | Scheduler for the refiner phase |
| **denoise_refiner** | Denoise amount for the refiner phase (0.0 to 1.0) |

### Example

With these settings:
- **steps**: 20
- **refiner_at_step**: 12

The node will:
1. Run **12 steps** with the main sampler/scheduler
2. Run **8 steps** with the refiner sampler/scheduler

### Refiner Denoise Control

The `denoise_refiner` parameter controls how much the refiner phase affects the image:

| Value | Effect |
|-------|--------|
| `0.0` | Refiner is skipped (only main sampler runs) |
| `0.5` | Refiner applies 50% denoising |
| `1.0` | Refiner applies full denoising |

## File Structure

```
KsamplerDF/
├── __init__.py
├── nodes.py
└── README.md
```

## Requirements

- ComfyUI (latest version recommended)
- No additional dependencies required

## License


MIT License - Feel free to use and modify as needed.


