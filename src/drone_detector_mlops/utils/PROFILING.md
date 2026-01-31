# PyTorch Training Profiler

This project includes optional performance profiling using PyTorch Profiler to analyze training runtime and memory behavior. Enable profiling with the `--profile` flag to record training traces and generate console summaries and TensorBoard visualizations.

## Usage

Create a .env file in the project root:

echo "WANDB_API_KEY=your_wandb_api_key_here" > .env

Get your W&B API key:

Go to wandb.ai/authorize (or Settings → API keys in your W&B account)
Copy your API key and paste it in the .env file

# For local runs

MODE=local

```bash
uv invoke run train --profile
```

View results:
- **Console**: Summary table prints automatically after training
- **TensorBoard**: `tensorboard --logdir=profiler-<timestamp>`

# For Cloud runs

MODE=cloud

```bash
uv invoke run cloud-train --profile
```


## Implementation

The profiler uses a small wrapper around `torch.profiler.profile` configured to record:
- CPU activity, memory usage, tensor shapes, and stack traces
- Clean traces: 1 wait step, 1 warmup step, 3 active steps
- One profiler step per training batch (forward pass + backward pass + optimizer update)
