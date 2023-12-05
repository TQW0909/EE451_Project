#!/bin/bash
#SBATCH --job-name=ABC_SERIAL_RUN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=epyc-64
#SBATCH --constraint=epyc-7513
#SBATCH --time=01:00:00
#SBATCH --output=ABC_SERIAL.out
#SBATCH --error=ABC_SERIAL.err

module purge
module load gcc/11.3.0

./original
