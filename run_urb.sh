#!/bin/bash
#SBATCH --job-name=URB
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=student

cd $HOME/URB
source .venv/bin/activate
singularity shell --nv -B /shared/sets/datasets:/dataset sumo-urb.sif
python scripts/ippo_torchrl.py --id saint_0 --alg-conf config5 --task-conf config5 --net ingolstadt_custom --env-seed 42 --torch-seed 0