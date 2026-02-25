from pathlib import Path
import torch

# Import modelclass from M3 (same architecture as in K2)
from src.ml.model import SimpleCNN
from src.ml.config import load_config

def main():
    cfg = load_config("params.yaml")

    weights_path = Path("data/models/model_weights.pt")
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Could not find weights:{weights_path}. "
            "Paste in artifacts/model_weights.pt from K2."
        )
    
    out_path = Path(cfg.service.model.path) # ex: data/models/model.ts.pt
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build the mode (must match with the architecture from K2)
    num_classes = len(cfg.service.labels)
    model = SimpleCNN(num_classes=num_classes)

    state = torch.load(weights_path, map_location="mps")
    model.load_state_dict(state)
    model.eval()

    c, h, w = cfg.service.input.channels, cfg.service.input.height, cfg.service.input.width
    example = torch.rand(1, c, h, w)

    scripted = torch.jit.trace(model, example)
    scripted.save(str(out_path))
    print(f"Export done: {out_path}")

    if __name__ == "__main__":
        main()