#!/bin/bash
#SBATCH --job-name=ABC_CPU_RUN
#SBATCH --partition=epyc-64
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=ABC_CPU.out
#SBATCH --error=ABC_CPU.err

module purge

module load gcc/11.3.0

./cpu

