#!/bin/bash
#SBATCH --job-name=mvu-mvu
#SBATCH --mem=30G # ram
#SBATCH --mincpus=3
#SBATCH --cpus-per-task=3
#SBATCH --output=logs/job.%A.mvu.out # %a


#SBATCH --gres=shard:0
#SBATCH --time=24:00:00

source .venv/bin/activate
python code/launcher.py --paper mvu --measure --threaded