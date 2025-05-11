#!/bin/bash
#SBATCH --job-name=image_classification    # create a short name for your job
#SBATCH -J 'joint-encode-sentences-image'
#SBATCH -o logfiles/conv-mod-%j.out
#SBATCH --error=logfiles/conv-modb%j.err
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=256G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:20:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=sm5607@princeton.edu
module purge
module load anaconda3/2024.2
conda activate env-RL

ENV_TYPE="gym"
ENV_NAME="cartpole/swingup"

python MRQ.py --env_type "$ENV_NAME" --env_name "$ENV_TYPE" 

