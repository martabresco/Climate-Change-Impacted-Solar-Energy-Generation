#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=CNRM-CM6-1
#SBATCH --nodes=1
#SBATCH --mail-user=s233224@dtu.dk
#SBATCH --mail-type=END
#SBATCH --exclusive=user
#SBATCH --partition=rome,workq

cd /groups/FutureWind/SFCRAD/CNRM-CM6-1/historical/r1i1p1f2

bash /groups/FutureWind/SFCRAD/CNRM-CM6-1/historical/r1i1p1f2/wget_script_2025-3-1_11-39-21.sh -H
