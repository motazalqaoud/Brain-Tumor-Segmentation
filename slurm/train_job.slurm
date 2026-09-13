#!/bin/bash
#SBATCH --job-name=brats-3d-attnunet
#SBATCH --partition=gpu                  # Confirmed: ODU's Wahab/Turing clusters name their
                                          #   CUDA-enabled partition "gpu" (per wiki.hpc.odu.edu/slurm)
#SBATCH --gres=gpu:1                     # 1 GPU. Wahab's GPU nodes are NVIDIA V100s.
                                          #   Bump to gpu:2 if you want to try multi-GPU (see note at bottom)
#SBATCH --cpus-per-task=8                # for DataLoader workers -- match --num_workers below
#SBATCH --mem=64G                        # 3D volumes + patch caching are memory-hungry on the CPU side too
#SBATCH --time=24:00:00                  # Check `sacctmgr show qos` or ask rcc@odu.edu for your max walltime;
                                          #   if it's shorter than this, see the "resuming" note below
#SBATCH --output=logs/%x_%j.out          # %x=job name, %j=job id
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL             # optional: email when the job finishes/fails
#SBATCH --mail-user=your_email@odu.edu   # CHANGE ME

# ============================================================================
# Brain tumor segmentation (3D Attention U-Net, MONAI) -- SLURM training job.
# Written for ODU Research Computing (Wahab/Turing, Open OnDemand + SLURM).
#
# Before first use:
#   1. Fill in your @odu.edu email above.
#   2. Confirm partition/GPU details with `sinfo -o "%P %G %m"` -- the "gpu"
#      partition name above is confirmed from ODU's own SLURM docs
#      (wiki.hpc.odu.edu/slurm), but cluster configs do change over time.
#   3. If you have an old working job script (e.g. job_script_gpu.sh from a
#      prior job), diff its module load lines against step 4 below -- module
#      names/versions can drift, and your own prior working script is the
#      most reliable source for exact current module names.
#   4. One-time environment setup (run these interactively, NOT in this script):
#        module avail cuda                  # find the exact available version
#        module load cuda/<version>          # e.g. module load cuda/12.1
#        module load anaconda3               # or whatever `module avail anaconda` shows
#        conda create -n brats python=3.10 -y
#        conda activate brats
#        pip install -r requirements.txt
#   5. One-time dataset download (also interactive -- compute nodes on some
#      clusters lack internet access, so run this on the login node or via
#      an Open OnDemand "Open in Terminal" session, not inside this SLURM job):
#        python data_prep.py --root_dir /scratch/$USER/brats_data
#
# Submit with:  sbatch slurm/train_job.slurm
# Monitor with: squeue -u $USER
# Cancel with:  scancel <job_id>
# ============================================================================

set -euo pipefail

# ---- Environment ----
module purge
module load cuda/12.1                    # CHANGE ME if `module avail cuda` shows a different version
module load anaconda3                    # CHANGE ME if your module name differs
source activate brats                    # CHANGE ME if you named your conda env something else

# ---- Paths (use scratch storage for data -- it's large and I/O-heavy;
#       home directories on most HPC systems have small quotas) ----
DATA_DIR=/scratch/$USER/brats_data/Task01_BrainTumour   # CHANGE ME
OUTPUT_DIR=/scratch/$USER/brats_checkpoints/$SLURM_JOB_ID
mkdir -p "$OUTPUT_DIR" logs

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Data dir: $DATA_DIR"
echo "Output dir: $OUTPUT_DIR"

# ---- Train ----
# patch_size/batch_size are left unset on purpose: train.py auto-detects them
# from the GPU's memory (see src/dataset.py recommend_patch_and_batch_size),
# since we don't know in advance whether SLURM lands this on a V100 or A100
# node. Override with --patch_size / --batch_size once you know your node's
# GPU and want to tune for max throughput instead of the safe default.
python src/train.py \
    --data_dir "$DATA_DIR" \
    --epochs 150 \
    --output_dir "$OUTPUT_DIR" \
    --num_workers "$SLURM_CPUS_PER_TASK" \
    --val_interval 5

echo "Job finished: $(date)"
echo "Best checkpoint: $OUTPUT_DIR/best_model.pth"
echo ""
echo "Next steps:"
echo "  1. Evaluate: python src/evaluate.py --data_dir $DATA_DIR --checkpoint $OUTPUT_DIR/best_model.pth"
echo "  2. Copy $OUTPUT_DIR/best_model.pth to checkpoints/best_model.pth in your local repo"
echo "     and to the Hugging Face Space to switch it out of demo mode."

# ============================================================================
# NOTES
#
# Walltime / resuming: if your cluster's max job walltime is shorter than the
# full training run needs, submit this as a job array or add a resume flag to
# train.py that loads the last checkpoint and continues -- ask me and I'll add
# it; it's a small change (load optimizer+scheduler state alongside the model
# weights, resume the epoch counter) that isn't in the current script since I
# didn't want to guess at your specific walltime limit.
#
# Multi-GPU: the current train.py is single-GPU. If your allocation gives you
# multiple GPUs and you want to use them, the straightforward path is
# torch.nn.parallel.DistributedDataParallel -- also not included here since it
# changes the training script's structure meaningfully; ask if you want it.
#
# If `module load cuda/12.1` fails: run `module avail cuda` on your cluster to
# see what's actually installed and swap in the right version/name. The exact
# module names are specific to ODU's HPC configuration, which isn't something
# I have visibility into from here.
# ============================================================================