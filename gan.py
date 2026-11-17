"""Train a DCGAN on a style subset of Kaggle WikiArt and save reproducible samples."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import save_image


STYLES = ("Abstract_Expressionism", "Color_Field_Painting", "Minimalism")
IMAGE_SIZE = 64


class Generator(nn.Sequential):
    def __init__(self, latent_dim=100, width=32, upsampling="transpose"):
        layers = [
            nn.ConvTranspose2d(latent_dim, width * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(width * 8), nn.ReLU(True),
        ]
        channels = width * 8
        for index, output in enumerate((width * 4, width * 2, width, 3)):
            is_rgb = index == 3
            if upsampling == "resize":
                layers.extend([
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(channels, output, 3, padding=1, bias=is_rgb),
                ])
            else:
                layers.append(nn.ConvTranspose2d(channels, output, 4, 2, 1, bias=is_rgb))
            if is_rgb:
                layers.append(nn.Tanh())
            else:
                layers.extend([nn.BatchNorm2d(output), nn.ReLU(True)])
            channels = output
        super().__init__(*layers)


class Discriminator(nn.Sequential):
    def __init__(self, width=32):
        super().__init__(
            nn.Conv2d(3, width, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(width, width * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(width * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(width * 2, width * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(width * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(width * 4, width * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(width * 8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(width * 8, 1, 4),
            nn.Flatten(start_dim=0),
        )


def initialize(layer):
    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(layer.weight, mean=0, std=0.02)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    elif isinstance(layer, nn.BatchNorm2d):
        nn.init.normal_(layer.weight, mean=1, std=0.02)
        nn.init.zeros_(layer.bias)


def select_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_image(path):
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        return rgb.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)


class Paintings(Dataset):
    def __init__(self, root, files, augment=False):
        self.root = root
        self.files = files
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        x = pil_to_tensor(read_image(self.root / self.files[index]))
        if self.augment and torch.rand(()).item() < 0.5:
            x = x.flip(-1)
        return x.float().div(127.5).sub(1)


def near_duplicate_indices(images):
    """Find later 64×64 RGB images with mean absolute difference at most 0.35/255."""
    if len(images) < 2:
        return set()

    pixels = np.stack(images)
    count, height, width, channels = pixels.shape
    limit = int(0.35 * height * width * channels)
    block_size = 8
    blocks = pixels.reshape(
        count, height // block_size, block_size, width // block_size, block_size, channels
    )
    block_sums = blocks.sum(axis=(2, 4), dtype=np.int32).reshape(count, -1)

    # Absolute differences of image and block sums bound the full pixel distance below.
    totals = block_sums.sum(axis=1)
    order = np.argsort(totals, kind="stable")
    sorted_totals = totals[order]
    lower = np.searchsorted(sorted_totals, totals - limit, side="left")
    upper = np.searchsorted(sorted_totals, totals + limit, side="right")

    duplicates = set()
    for i, image in enumerate(pixels):
        candidates = order[lower[i]:upper[i]]
        earlier = candidates[candidates < i]
        coarse_distance = np.abs(block_sums[earlier] - block_sums[i]).sum(axis=1)
        rgb = image.astype(np.int16)
        # Check previously discarded images too, since they may have near duplicates of their own.
        for j in earlier[coarse_distance <= limit]:
            if np.abs(rgb - pixels[j]).sum() <= limit:
                duplicates.add(i)
                break
    return duplicates


def make_split(root, holdout, seed, max_train):
    rng = random.Random(seed)
    entries = []
    images = []
    seen = set()
    duplicates = 0
    for style in STYLES:
        for path in sorted((root / style).rglob("*")):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            image = read_image(path)
            raw = image.tobytes()
            digest = hashlib.sha256(raw).digest()
            if digest in seen:
                duplicates += 1
            else:
                seen.add(digest)
                entries.append((style, path.relative_to(root).as_posix()))
                images.append(np.frombuffer(raw, dtype=np.uint8).reshape(IMAGE_SIZE, IMAGE_SIZE, 3))

    near_duplicates = near_duplicate_indices(images)
    sources = {style: [] for style in STYLES}
    for i, (style, file) in enumerate(entries):
        if i not in near_duplicates:
            sources[style].append(file)
    for style, files in sources.items():
        if not files:
            raise ValueError(f"No unique images for {style} in {root} after deduplication.")
        rng.shuffle(files)
    total = sum(len(files) for files in sources.values())
    if not 2 <= holdout < total:
        raise ValueError(f"Holdout size must be between 2 and {total - 1}.")
    # Proportional holdout with largest-remainder rounding.
    shares = {style: holdout * len(files) / total for style, files in sources.items()}
    counts = {style: int(share) for style, share in shares.items()}
    remainder = holdout - sum(counts.values())
    for style in sorted(shares, key=lambda name: -(shares[name] - counts[name]))[:remainder]:
        counts[style] += 1
    train, test = [], []
    for style, files in sources.items():
        test.extend(files[:counts[style]])
        train.extend(files[counts[style]:])
    rng.shuffle(train)
    rng.shuffle(test)
    if max_train:
        train = train[:max_train]
    return {
        "train": train, "holdout": test, "exact_duplicates_removed": duplicates,
        "near_duplicates_removed": len(near_duplicates),
    }


@torch.inference_mode()
def preview(generator, noise, path):
    generator.eval()
    images = generator(noise).add(1).div(2).clamp(0, 1).cpu()
    save_image(images, path, nrow=8, padding=2)
    generator.train()


def plot_losses(history, path):
    fig, ax = plt.subplots(figsize=(5.5, 2.3), layout="constrained")
    epochs = [item["epoch"] for item in history]
    ax.plot(epochs, [item["loss_g"] for item in history], label="Generator", linewidth=1.5)
    ax.plot(epochs, [item["loss_d"] for item in history], label="Discriminator", linewidth=1.5)
    ax.set(xlabel="Epoch", ylabel="Mean loss", xlim=(0.8, max(2, epochs[-1] + 0.2)))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=230)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--holdout", type=int, default=512)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--upsampling", choices=("transpose", "resize"), default="transpose")
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--max-train", type=int, default=None)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.holdout, args.width, args.latent_dim, args.threads) < 1:
        parser.error("Integer settings must be positive.")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Choose an empty output directory.")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    split = make_split(args.data, args.holdout, args.seed, args.max_train)
    (args.out / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    config = {
        "epochs": args.epochs, "batch_size": args.batch_size, "latent_dim": args.latent_dim,
        "width": args.width, "lr": args.lr, "betas": [0.5, 0.999], "seed": args.seed,
        "device": str(device), "image_size": IMAGE_SIZE, "train_images": len(split["train"]),
        "holdout_images": len(split["holdout"]), "styles": list(STYLES),
        "exact_duplicates_removed": split["exact_duplicates_removed"],
        "near_duplicates_removed": split["near_duplicates_removed"],
        "upsampling": args.upsampling,
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    dataset = Paintings(args.data, split["train"], augment=True)
    batch_size = min(args.batch_size, len(dataset))
    batches = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                         generator=torch.Generator().manual_seed(args.seed))
    print(f"Training on {device}: {len(dataset)} train, {args.holdout} holdout, "
          f"{len(batches)} batches per epoch, {args.epochs} epochs.", flush=True)
    g = Generator(args.latent_dim, args.width, args.upsampling).to(device).apply(initialize)
    d = Discriminator(args.width).to(device).apply(initialize)
    opt_g = torch.optim.Adam(g.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(d.parameters(), lr=args.lr, betas=(0.5, 0.999))
    loss = nn.BCEWithLogitsLoss()
    fixed = torch.randn(24, args.latent_dim, 1, 1,
                        generator=torch.Generator().manual_seed(args.seed + 1)).to(device)
    torch.save({"generator": g.state_dict(), "config": config, "epoch": 0},
               args.out / "initial.pt")
    preview(g, fixed, args.out / "samples_000.png")
    history = []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        total_g = total_d = 0.0
        for step, real in enumerate(batches, 1):
            real = real.to(device)
            n = len(real)
            opt_d.zero_grad(set_to_none=True)
            fake = g(torch.randn(n, args.latent_dim, 1, 1, device=device))
            # Keep the fake-only BatchNorm context identical between the two updates.
            real_logits = d(real)
            fake_logits = d(fake.detach())
            loss_d = (loss(real_logits, torch.ones_like(real_logits))
                      + loss(fake_logits, torch.zeros_like(fake_logits)))
            loss_d.backward()
            opt_d.step()

            opt_g.zero_grad(set_to_none=True)
            d.requires_grad_(False)
            fake_logits = d(fake)
            loss_g = loss(fake_logits, torch.ones_like(fake_logits))
            loss_g.backward()
            opt_g.step()
            d.requires_grad_(True)
            total_g += loss_g.item() * n
            total_d += loss_d.item() * n
            if step == 1 or step % 20 == 0:
                print(f"Epoch {epoch}/{args.epochs}, step {step}/{len(batches)}: "
                      f"G {loss_g.item():.3f}, D {loss_d.item():.3f}", flush=True)
        row = {"epoch": epoch, "loss_g": total_g / len(dataset), "loss_d": total_d / len(dataset),
               "elapsed_seconds": round(time.monotonic() - started, 1)}
        history.append(row)
        with (args.out / "history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=row)
            writer.writeheader()
            writer.writerows(history)
        preview(g, fixed, args.out / f"samples_{epoch:03d}.png")
        state = {"generator": g.state_dict(), "discriminator": d.state_dict(),
                 "optimizer_g": opt_g.state_dict(), "optimizer_d": opt_d.state_dict(),
                 "config": config, "epoch": epoch}
        torch.save(state, args.out / "last.tmp")
        (args.out / "last.tmp").replace(args.out / "last.pt")
        plot_losses(history, args.out / "loss_curves.png")
        print(row, flush=True)


if __name__ == "__main__":
    main()
