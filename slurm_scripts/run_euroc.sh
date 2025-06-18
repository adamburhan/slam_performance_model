#!/bin/bash
#SBATCH --time=50:00:00
#SBATCH --mem=50G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output="/network/scratch/a/adam.burhan/slurm-%j.out"

module load singularity

# 1. copy container to the local disk
rsync -avz $SCRATCH/pytorch_final.sif $SLURM_TMPDIR

# 2. copy dataset on the compute node
rsync -avz $SCRATCH/datasets/ood_slam.tar.gz $SLURM_TMPDIR

tar xzf $SLURM_TMPDIR/ood_slam.tar.gz -C $SLURM_TMPDIR/

# 3. Executing the code with singularity
singularity exec --nv \
        -H $HOME/projects/ood_slam/slam_performance_model:/home \
        -B $SLURM_TMPDIR:/dataset/ \
        -B $SLURM_TMPDIR:/tmp_log/ \
        -B $SCRATCH:/final_log \
        $SLURM_TMPDIR/pytorch_final.sif \
        python /home/main.py --config /home/configs/euroc_test.yaml

# 4. Copy what needs to be saved on $SCRATCH
rsync -avz $SLURM_TMPDIR/final_log/ $SCRATCH