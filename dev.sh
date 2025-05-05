#!/bin/bash
#SBATCH --job-name=mvu-dev
#SBATCH --mem=100G # ram
#SBATCH --mincpus=1
#SBATCH --output=logs/job.%A.dev.out # %a


#SBATCH --gres=shard:0
#SBATCH --time=24:00:00

source .venv/bin/activate
python code/launcher.py --paper dev --threaded