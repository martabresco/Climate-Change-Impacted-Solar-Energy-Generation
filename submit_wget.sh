#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=CMCC-CM2-SR5
#SBATCH --nodes=1
#SBATCH --mail-user=s233224@dtu.dk
#SBATCH --mail-type=END
#SBATCH --exclusive=user
#SBATCH --partition=rome,workq

cd /groups/FutureWind/SFCRAD/CMCC-CM2-SR5/historical/r1i1p1f1

bash /groups/FutureWind/SFCRAD/CMCC-CM2-SR5/wget_script_2025-2-26_13-34-33.sh -H
