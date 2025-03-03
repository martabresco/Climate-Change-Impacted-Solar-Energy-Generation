#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=UKESM1-0-LL
#SBATCH --nodes=1
#SBATCH --mail-user=s233224@dtu.dk
#SBATCH --mail-type=END
#SBATCH --exclusive=user
#SBATCH --partition=rome,workq

cd /groups/FutureWind/SFCRAD/UKESM1-0-LL/historical/r1i1p1f2

bash /groups/FutureWind/SFCRAD/UKESM1-0-LL/historical/r1i1p1f2/wget_script_2025-3-3_18-50-51.sh -H
