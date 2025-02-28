#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=ACCESS-CM2
#SBATCH --nodes=1
#SBATCH --mail-user=s233224@dtu.dk
#SBATCH --mail-type=END
#SBATCH --exclusive=user
#SBATCH --partition=rome,workq

cd /groups/FutureWind/SFCRAD/ACCESS-CM2/ssp585/r1i1p1f1

bash /work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation/wget_script_2025-2-26_20-45-39.sh -H
