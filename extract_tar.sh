#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=sarah_data_extract
#SBATCH --nodes=1
#SBATCH --mail-user=s233224@dtu.dk
#SBATCH --mail-type=END
#SBATCH --exclusive=user
#SBATCH --partition=rome,workq

# Extract the .tar file(s)
tar -xvf "/work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation/sarah_data.tar" -C /work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation/sarah_data/