# Create the models/tryon directory if it doesn't exist
import os
from pathlib import Path

# Get the project root
project_root = Path(__file__).parent.parent.parent
models_dir = project_root / "models" / "tryon"
models_dir.mkdir(parents=True, exist_ok=True)

print(f"Created models directory at: {models_dir}")