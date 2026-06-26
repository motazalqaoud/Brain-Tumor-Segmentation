"""
run.py — Brain Tumor Segmentation Pipeline
===========================================
Single entry point for setup, training, inference, and results.

  python run.py setup          # download dataset, verify everything works
  python run.py train          # train the model (hardware auto-detected)
  python run.py predict        # run inference with the best checkpoint
  python run.py results        # generate prediction figures for all tumor types

Run  python run.py <command> --help  for options.
"""

import argparse
import subprocess
import sys
import os
import json
from pathlib import Path


ROOT = Path(__file__).parent
SCRIPTS = ROOT / "scripts"
CONFIGS = ROOT / "configs"
CHECKPOINTS = ROOT / "checkpoints"
RESULTS = ROOT / "results"
DATA_ROOT = ROOT / "data" / "raw" / "Images_"


# ── helpers ──────────────────────────────────────────────────────────────────

def run(cmd: list, **kwargs):
    """Run a subprocess command, streaming output live."""
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        sys.exit(result.returncode)


def detect_hardware() -> tuple[str, str]:
    """Return (device, config_file) based on available hardware."""
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            name = torch.cuda.get_device_name(0)
            if vram_gb >= 14:
                return "cuda", str(CONFIGS / "gpu_16gb.json"), name, vram_gb
            else:
                return "cuda", str(CONFIGS / "gpu_8gb.json"), name, vram_gb
        else:
            import platform
            return "cpu", str(CONFIGS / "cpu.json"), platform.processor(), None
    except ImportError:
        return "cpu", str(CONFIGS / "cpu.json"), "unknown", None


def find_best_checkpoint() -> Path | None:
    """Return the checkpoint with the highest Dice in its filename."""
    if not CHECKPOINTS.exists():
        return None
    candidates = sorted(
        CHECKPOINTS.glob("best_model_dice_*.pt"),
        key=lambda p: float(p.stem.split("_")[-1]),
        reverse=True
    )
    return candidates[0] if candidates else CHECKPOINTS / "checkpoint_latest.pt"


def find_sample_image(tumor_type: str) -> tuple[Path, Path] | tuple[None, None]:
    """Find one image+mask pair for a given tumor type."""
    tumor_dir = DATA_ROOT / tumor_type
    if not tumor_dir.exists():
        return None, None
    for modality_dir in tumor_dir.iterdir():
        if not modality_dir.is_dir():
            continue
        for subtype_dir in modality_dir.iterdir():
            if not subtype_dir.is_dir():
                continue
            for img_file in subtype_dir.iterdir():
                if img_file.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    if '_mask' in img_file.name or '_bbox' in img_file.name:
                        continue
                    mask = img_file.parent / f"{img_file.stem}_mask_consensus.png"
                    if mask.exists():
                        return img_file, mask
    return None, None


def print_banner(title: str):
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


# ── commands ─────────────────────────────────────────────────────────────────

def cmd_setup(args):
    print_banner("Setup — Brain Tumor Segmentation")

    # 1. Check Python
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 8):
        print(f"ERROR: Python 3.8+ required (you have {major}.{minor})")
        sys.exit(1)
    print(f"✓ Python {major}.{minor}")

    # 2. Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✓ CUDA available — {torch.cuda.get_device_name(0)}")
        else:
            print("  CUDA not available — will train on CPU (slower)")
    except ImportError:
        print("  PyTorch not installed. Running: pip install -r requirements.txt")
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    # 3. Check Kaggle API
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("\nKaggle API key not found.")
        print("To download the dataset you need a free Kaggle account:")
        print("  1. Go to https://www.kaggle.com → Account → API → Create New Token")
        print("  2. Move the downloaded kaggle.json to: ~/.kaggle/kaggle.json")
        print("  3. Run: chmod 600 ~/.kaggle/kaggle.json")
        print("  4. Re-run: python run.py setup\n")
        sys.exit(1)
    print("✓ Kaggle API key found")

    # 4. Download dataset
    if DATA_ROOT.exists() and any(DATA_ROOT.iterdir()):
        print(f"✓ Dataset already present at {DATA_ROOT}")
    else:
        print("\nDownloading Brain Tumor 12K dataset (~1.2GB)...")
        run([sys.executable, "-m", "kaggle", "datasets", "download",
             "-d", "fernando2rad/brain-tumor-12k-mri-images-w-masks-meta-and-bbox",
             "-p", str(ROOT / "data" / "raw")])
        import zipfile
        zips = list((ROOT / "data" / "raw").glob("*.zip"))
        if zips:
            print(f"\nExtracting {zips[0].name}...")
            with zipfile.ZipFile(zips[0]) as zf:
                zf.extractall(ROOT / "data" / "raw")
            zips[0].unlink()
        print(f"✓ Dataset extracted to {ROOT / 'data' / 'raw'}")

    # 5. Verify
    print("\nVerifying dataset and model forward pass...")
    run([sys.executable, str(SCRIPTS / "test_model.py"),
         "--data-root", str(DATA_ROOT)])

    print("\n✓ Setup complete. Run:  python run.py train")


def cmd_train(args):
    print_banner("Train — 3D Attention U-Net")

    # Verify dataset exists
    if not DATA_ROOT.exists() or not any(DATA_ROOT.iterdir()):
        print("Dataset not found. Run first:  python run.py setup")
        sys.exit(1)

    # Auto-detect hardware
    result = detect_hardware()
    device, config_file = result[0], result[1]
    hw_name = result[2]
    vram = result[3]

    print(f"Hardware detected: {hw_name}")
    if vram:
        print(f"VRAM: {vram:.1f} GB")
    print(f"Config: {Path(config_file).name}")

    # Load and show config
    with open(config_file) as f:
        cfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    print("\nTraining settings:")
    for k, v in cfg.items():
        print(f"  {k:20s} {v}")

    # Time estimate
    epoch_min = 7 if device == "cpu" else 2
    total_min = epoch_min * cfg.get("epochs", 50)
    hours = total_min // 60
    mins = total_min % 60
    print(f"\nEstimated time: ~{hours}h {mins}m on this hardware")

    if not args.yes:
        answer = input("\nStart training? [y/n]: ").strip().lower()
        if answer != "y":
            print("Cancelled.")
            return

    # Override config with user flags if provided
    cmd = [
        sys.executable, str(SCRIPTS / "train3d.py"),
        "--config", config_file,
        "--data-root", str(DATA_ROOT),
    ]
    if args.epochs:
        cmd += ["--epochs", str(args.epochs)]
    if args.resume or (CHECKPOINTS / "checkpoint_latest.pt").exists():
        resume = args.resume or str(CHECKPOINTS / "checkpoint_latest.pt")
        cmd += ["--resume", resume]
        print(f"\nResuming from: {resume}")

    run(cmd)
    print("\n✓ Training complete. Run:  python run.py results")


def cmd_predict(args):
    print_banner("Predict — Run Inference")

    checkpoint = args.checkpoint or find_best_checkpoint()
    if checkpoint is None:
        print("No checkpoint found. Run first:  python run.py train")
        sys.exit(1)
    print(f"Checkpoint: {checkpoint}")

    cmd = [
        sys.executable, str(SCRIPTS / "predict3d.py"),
        "--checkpoint", str(checkpoint),
        "--out", args.out,
    ]
    if args.image:
        cmd += ["--image", args.image]
        if args.mask:
            cmd += ["--mask", args.mask]
    else:
        # Try to find a real image from the dataset
        for tumor_type in ["Glioma", "Meningioma", "Pituitary"]:
            img, mask = find_sample_image(tumor_type)
            if img:
                cmd += ["--image", str(img), "--mask", str(mask)]
                print(f"Using: {img.name} ({tumor_type})")
                break
        else:
            print("No dataset found — running with synthetic demo input.")

    run(cmd)
    print(f"\n✓ Saved → {args.out}")


def cmd_results(args):
    print_banner("Results — Generate All Figures")

    checkpoint = args.checkpoint or find_best_checkpoint()
    if checkpoint is None:
        print("No checkpoint found. Run first:  python run.py train")
        sys.exit(1)
    print(f"Checkpoint: {checkpoint}")

    RESULTS.mkdir(exist_ok=True)

    generated = []
    for tumor_type in ["Glioma", "Meningioma", "Pituitary"]:
        img, mask = find_sample_image(tumor_type)
        if img is None:
            print(f"  WARNING: No {tumor_type} sample found — skipping")
            continue

        out = RESULTS / f"{tumor_type.lower()}_prediction.png"
        print(f"\nGenerating {tumor_type} prediction...")
        run([
            sys.executable, str(SCRIPTS / "predict3d.py"),
            "--checkpoint", str(checkpoint),
            "--image", str(img),
            "--mask", str(mask),
            "--out", str(out),
        ])
        generated.append((tumor_type, out))

    # Copy training curves if they exist
    curves_src = ROOT / "visualizations" / "training_curves.png"
    if curves_src.exists():
        import shutil
        shutil.copy(curves_src, RESULTS / "training_curves.png")
        print(f"\n✓ Copied training curves → results/training_curves.png")

    print(f"\n✓ Results saved to results/")
    for tumor_type, path in generated:
        print(f"   {tumor_type:12s} → {path.name}")
    if (RESULTS / "training_curves.png").exists():
        print(f"   Curves       → training_curves.png")
    print("\nNext: commit results/ and update the README with these figures.")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Brain Tumor Segmentation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  setup    Download dataset, verify installation
  train    Train the model (hardware auto-detected)
  predict  Run inference with the best checkpoint
  results  Generate prediction figures for all tumor types

Examples:
  python run.py setup
  python run.py train
  python run.py train --epochs 100
  python run.py predict --image path/to/image.jpg --mask path/to/mask.png
  python run.py results
        """
    )
    sub = parser.add_subparsers(dest="command")

    # setup
    sub.add_parser("setup", help="Download dataset and verify installation")

    # train
    p_train = sub.add_parser("train", help="Train the model")
    p_train.add_argument("--epochs", type=int, default=None,
                         help="Override number of epochs from config")
    p_train.add_argument("--resume", type=str, default=None,
                         help="Path to checkpoint to resume from")
    p_train.add_argument("-y", "--yes", action="store_true",
                         help="Skip confirmation prompt")

    # predict
    p_pred = sub.add_parser("predict", help="Run inference on an image")
    p_pred.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (auto-detected if omitted)")
    p_pred.add_argument("--image", type=str, default=None,
                        help="Path to input MRI image (JPG/PNG)")
    p_pred.add_argument("--mask", type=str, default=None,
                        help="Path to ground truth mask PNG (optional)")
    p_pred.add_argument("--out", type=str, default="prediction.png",
                        help="Output figure path (default: prediction.png)")

    # results
    p_res = sub.add_parser("results", help="Generate all result figures")
    p_res.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint (auto-detected if omitted)")

    args = parser.parse_args()

    if args.command == "setup":
        cmd_setup(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "results":
        cmd_results(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
