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


def check_filenames(value, label):
    if not isinstance(value, list) or any(
        not isinstance(name, str) or any(part in {"", ".", ".."} for part in name.split("/"))
        or "\\" in name for name in value
    ):
        raise ValueError(f"{label} must contain a JSON list of relative filenames")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} contains duplicate filenames")
    return value


def read_holdout(run):
    path = run / "split.json"
    split = json.loads(path.read_text())
    return check_filenames(split.get("holdout") if isinstance(split, dict) else None, path)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path, action="append",
                        help="repeat to evaluate multiple runs with a shared ordered holdout")
    parser.add_argument("--last-only", action="store_true", help="skip the initial checkpoint")
    parser.add_argument("--exclude-holdout", type=Path,
                        help="JSON list of relative holdout filenames to omit")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    if min(args.batch_size, args.threads) < 1:
        parser.error("--batch-size and --threads must be positive")
    if len({run.resolve() for run in args.run}) != len(args.run):
        parser.error("--run directories must be distinct")
    try:
        holdout = read_holdout(args.run[0])
        for run in args.run[1:]:
            if read_holdout(run) != holdout:
                parser.error(f"{run} does not have the same ordered holdout as {args.run[0]}")
        excluded = []
        if args.exclude_holdout is not None:
            excluded = check_filenames(json.loads(args.exclude_holdout.read_text()), args.exclude_holdout)
            unknown = set(excluded).difference(holdout)
            if unknown:
                parser.error(f"Exclusion list contains a filename outside the holdout: {min(unknown)}")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    excluded_names = set(excluded)
    indices = [index for index, name in enumerate(holdout) if name not in excluded_names]
    if len(indices) < 2:
        parser.error("At least two holdout images must remain")

    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    dataset = Paintings(args.data, [holdout[index] for index in indices])
    batches = DataLoader(dataset, batch_size=args.batch_size)
    checkpoints = ("last",) if args.last_only else ("initial", "last")
    groups = {}
    latent_dim = None
    for run in args.run:
        groups[run] = {}
        for name in checkpoints:
            state = torch.load(run / f"{name}.pt", map_location="cpu", weights_only=True)
            config = state["config"]
            if latent_dim is None:
                latent_dim = config["latent_dim"]
            elif config["latent_dim"] != latent_dim:
                parser.error(f"{run / (name + '.pt')} has a different latent dimension")
            generator = Generator(latent_dim, config["width"], config.get("upsampling", "transpose"))
            generator.load_state_dict(state["generator"])
            groups[run][name] = {
                "generator": generator.to(device).eval(),
                "fid": FID(num_features=2048, feature_extractor=nn.Identity()),
                "is": InceptionScore(num_features=1000, feature_extractor=nn.Identity()),
                "epoch": state["epoch"],
            }

    backbone = Inception().to(device)
    rng = torch.Generator().manual_seed(args.seed)
    # Draw against the original holdout so filtering or re-batching keeps each image's latent.
    noise = torch.randn(len(holdout), latent_dim, 1, 1, generator=rng)[indices]
    completed = 0
    for real in batches:
        n = len(real)
        z = noise[completed:completed + n].to(device)
        real_features = backbone(real.to(device))[0].cpu()
        for run, metrics in groups.items():
            for name, bundle in metrics.items():
                fake = bundle["generator"](z)
                features, probs = backbone(fake)
                bundle["fid"].update((features.cpu(), real_features))
                bundle["is"].update(probs.cpu())
                if completed == 0:
                    save_image(fake.cpu().add(1).div(2), run / f"eval_{name}.png",
                               nrow=8, padding=2)
            if completed == 0:
                save_image(real.add(1).div(2), run / "eval_real.png", nrow=8, padding=2)
        completed += n
        if completed % 64 == 0 or completed == len(dataset):
            print(f"Inception features: {completed}/{len(dataset)} per set", flush=True)

    protocol = {
        "num_real": completed, "num_generated_per_checkpoint": completed, "seed": args.seed,
        "reference": "stratified held-out paintings excluded from GAN training",
        "original_holdout_count": len(holdout),
        "excluded_holdout": excluded,
        "evaluation_batch_size": args.batch_size,
        "evaluation_device": str(device),
        "fid_representation": "2048-dimensional pooled ImageNet Inception-v3 features",
        "is_representation": "1000-class ImageNet probabilities, one pooled set",
        "preprocessing": "64x64 RGB, denormalize, bilinear 299x299, ImageNet normalization",
        "torch": torch.__version__, "torchvision": torchvision.__version__, "ignite": ignite.__version__,
    }
    for run, metrics in groups.items():
        results = protocol.copy()
        for name, bundle in metrics.items():
            print(f"Computing Ignite FID and IS for {run / name}", flush=True)
            results[name] = {"epoch": bundle["epoch"], "fid": float(bundle["fid"].compute()),
                             "inception_score": float(bundle["is"].compute())}
        (run / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
