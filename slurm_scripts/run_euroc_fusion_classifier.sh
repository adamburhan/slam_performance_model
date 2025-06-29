#!/bin/bash
#SBATCH --time=50:00:00
#SBATCH --mem=50G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output="/network/scratch/a/adam.burhan/slurm-%A_%a.out"
#SBATCH --array=0-3

module load singularity

# image encoders array
encoders=("ImageEncoder" "TemporalImageEncoder" "LSTMImageEncoder" "TransformerImageEncoder")
ENCODER=${encoders[$SLURM_ARRAY_TASK_ID]}

exp_names=("fusion_img_3" "fusion_3dcnn_3" "fusion_lstm_3" "fusion_transformer_3")
EXP_NAME=${exp_names[$SLURM_ARRAY_TASK_ID]}

echo "Running experiment with encoder: $ENCODER, experiment name: $EXP_NAME"

# 1. copy container to the local disk
rsync -avz $SCRATCH/pytorch_final.sif $SLURM_TMPDIR

# 2. copy dataset on the compute node
rsync -avz $SCRATCH/datasets/ood_slam/ $SLURM_TMPDIR

echo "Contents of $SLURM_TMPDIR:"
ls -l $SLURM_TMPDIR

#tar xzf $SLURM_TMPDIR/ood_slam.tar.gz -C $SLURM_TMPDIR/

mkdir -p $SLURM_TMPDIR/tmp_log
mkdir -p $SLURM_TMPDIR/wandb

# 3. Executing the code with singularity
singularity exec --nv \
        --env WANDB_MODE="offline" \
        --env WANDB_API_KEY="65f9f067d5b47932d6cfaba52c346d7b3e435bf9" \
        --env WANDB_RUN_GROUP="fusion_classifier_comparison_EMD" \
        -H $HOME/projects/ood_slam/slam_performance_model:/home \
        -B $SLURM_TMPDIR:/dataset/ \
        -B $SLURM_TMPDIR/tmp_log:/tmp_log/ \
        -B $SLURM_TMPDIR/wandb:/wandb \
        $SLURM_TMPDIR/pytorch_final.sif \
        python /home/main.py --config /home/configs/fusion_classifier_${ENCODER}.yaml

# 4. Copy what needs to be saved on $SCRATCH
mkdir -p $SCRATCH/logs/${EXP_NAME}
mkdir -p $SCRATCH/wandb/${EXP_NAME}

rsync -avz $SLURM_TMPDIR/tmp_log/ $SCRATCH/logs/${EXP_NAME}
rsync -avz $SLURM_TMPDIR/wandb/ $SCRATCH/wandb/${EXP_NAME}