#!/bin/bash

#SBATCH -J ceal    # Job Name

## Leave these values as they are unless you know what you are doing
#SBATCH --ntasks=1             # Tasks
#SBATCH --nodes=1              # Number of nodes
#SBATCH --ntasks-per-core=2    # Number of processes per CPU code
#SBATCH --cpus-per-task=1      # Number of CPUs per task?
#SBATCH --gres=gpu:tesla:1     # Use 1 GPU per node

## Adjust to your needs
#SBATCH --mem=25GB             # Memory limit per node
#SBATCH --time=168:00:00       # Expected maximum run time
#SBATCH --partition=gpu        # This is needed to use a GPU

## Job-Status per Mail
#SBATCH --mail-type=ALL
#SBATCH --mail-user=t.tranthi@campus.tu-berlin.de

ulimit -u 512
cd src/
rm -d output_* -r -f

source activate ceal
python CEAL.py

