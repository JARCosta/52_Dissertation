#!/bin/bash
#SBATCH --job-name=mvu-eng_models
#SBATCH --mem=50G # ram
#SBATCH --mincpus=1
#SBATCH --output=logs/job.%A.eng.out # %a


#SBATCH --gres=shard:0
#SBATCH --time=24:00:00

source .venv/bin/activate
python code/launcher.py --paper eng --threaded