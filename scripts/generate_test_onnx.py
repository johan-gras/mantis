#!/usr/bin/env python3
"""Generate sample ONNX models for testing Mantis ONNX integration.

This script creates minimal ONNX models for testing the inference pipeline.
The models are simple enough to be fast but realistic enough for integration testing.
"""

import torch
import torch.nn as nn
import os


class SimpleMLP(nn.Module):
    """A simple MLP for regression (predicting a single output from features).

    This model takes 10 input features and produces a single output value,
    suitable for generating trading signals.
    """

    def __init__(self, input_size: int = 10, hidden_size: int = 32, output_size: int = 1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh(),  # Output between -1 and 1 for signal scaling
        )

    def forward(self, x):
        return self.network(x)


class MinimalModel(nn.Module):
    """Smallest possible model - single linear layer."""

    def __init__(self, input_size: int = 10, output_size: int = 1):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.activation = nn.Tanh()

    def forward(self, x):
        return self.activation(self.linear(x))


def export_model(model: nn.Module, path: str, input_size: int, batch_size: int = 1):
    """Export a PyTorch model to ONNX format."""
    import onnx

    model.eval()

    # Create dummy input
    dummy_input = torch.randn(batch_size, input_size)

    # Export to ONNX with opset 18 (well supported by ort 2.0)
    torch.onnx.export(
        model,
        dummy_input,
        path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        },
    )

    # Load the model
    onnx_model = onnx.load(path)

    # Convert external data to internal (single file)
    from onnx.external_data_helper import convert_model_to_external_data
    onnx.save_model(
        onnx_model,
        path,
        save_as_external_data=False,  # Force internal data
    )

    # Remove any external data file
    data_file = path + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)

    # Verify the model
    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)

    # Get file size
    size_kb = os.path.getsize(path) / 1024
    print(f"  Created: {path} ({size_kb:.1f} KB)")


def main():
    """Generate test ONNX models."""
    # Ensure data directory exists
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "models")
    os.makedirs(models_dir, exist_ok=True)

    print("Generating test ONNX models...")

    # 1. Simple MLP model (10 inputs, small enough to be fast)
    mlp = SimpleMLP(input_size=10, hidden_size=32, output_size=1)
    export_model(mlp, os.path.join(models_dir, "simple_mlp.onnx"), input_size=10)

    # 2. Minimal model (for fastest possible inference)
    minimal = MinimalModel(input_size=10, output_size=1)
    export_model(minimal, os.path.join(models_dir, "minimal.onnx"), input_size=10)

    # 3. Larger model for stress testing
    larger = SimpleMLP(input_size=20, hidden_size=64, output_size=1)
    export_model(larger, os.path.join(models_dir, "larger_mlp.onnx"), input_size=20)

    print("\nAll models generated successfully!")
    print(f"\nModels saved to: {models_dir}")


if __name__ == "__main__":
    main()
