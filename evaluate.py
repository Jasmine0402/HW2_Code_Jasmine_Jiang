"""Evaluate initialized and trained generators with Ignite FID and Inception Score."""
import argparse
import json
from pathlib import Path

import ignite
from ignite.metrics import FID, InceptionScore
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.utils import save_image

from gan import Generator, Paintings, select_device


class Inception(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        self.classifier = self.network.fc
        self.network.fc = nn.Identity()
        self.register_buffer("pixel_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("pixel_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        self.eval().requires_grad_(False)

    def forward(self, images):
        # GAN tensors are in [-1, 1]. Present real and generated RGB images identically.
        unit = images.add(1).div(2).clamp(0, 1)
        resized = F.interpolate(unit, (299, 299), mode="bilinear",
                                align_corners=False, antialias=True)
        features = self.network((resized - self.pixel_mean) / self.pixel_std)
        return features, self.classifier(features).softmax(dim=1)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    split = json.loads((args.run / "split.json").read_text())
    dataset = Paintings(args.data, split["holdout"])
    batches = DataLoader(dataset, batch_size=args.batch_size)
    backbone = Inception().to(device)
    metrics = {}
    generators = {}
    for name in ("initial", "last"):
        state = torch.load(args.run / f"{name}.pt", map_location="cpu", weights_only=True)
        config = state["config"]
        generator = Generator(config["latent_dim"], config["width"], config.get("upsampling", "transpose"))
        generator.load_state_dict(state["generator"])
        generators[name] = generator.to(device).eval()
        metrics[name] = {
            "fid": FID(num_features=2048, feature_extractor=nn.Identity()),
            "is": InceptionScore(num_features=1000, feature_extractor=nn.Identity()),
            "epoch": state["epoch"],
        }
    rng = torch.Generator().manual_seed(args.seed)
    completed = 0
    for real in batches:
        n = len(real)
        z = torch.randn(n, config["latent_dim"], 1, 1, generator=rng).to(device)
        real_features, _ = backbone(real.to(device))
        for name, generator in generators.items():
            fake = generator(z)
            features, probs = backbone(fake)
            metrics[name]["fid"].update((features.cpu(), real_features.cpu()))
            metrics[name]["is"].update(probs.cpu())
            if completed == 0:
                save_image(fake.cpu().add(1).div(2), args.run / f"eval_{name}.png",
                           nrow=8, padding=2)
        if completed == 0:
            save_image(real.add(1).div(2), args.run / "eval_real.png", nrow=8, padding=2)
        completed += n
        if completed % 64 == 0 or completed == len(dataset):
            print(f"Inception features: {completed}/{len(dataset)} per set", flush=True)
    results = {
        "num_real": completed, "num_generated_per_checkpoint": completed, "seed": args.seed,
        "reference": "stratified held-out paintings excluded from GAN training",
        "fid_representation": "2048-dimensional pooled ImageNet Inception-v3 features",
        "is_representation": "1000-class ImageNet probabilities, one pooled set",
        "preprocessing": "64x64 RGB, denormalize, bilinear 299x299, ImageNet normalization",
        "torchvision": torchvision.__version__, "ignite": ignite.__version__,
    }
    for name, bundle in metrics.items():
        print(f"Computing Ignite FID and IS for {name}", flush=True)
        results[name] = {"epoch": bundle["epoch"], "fid": float(bundle["fid"].compute()),
                         "inception_score": float(bundle["is"].compute())}
    (args.run / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
