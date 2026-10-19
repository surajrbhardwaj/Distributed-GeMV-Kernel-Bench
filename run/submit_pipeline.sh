#!/bin/bash -l
#SBATCH --job-name=PipelineBenchmark
#SBATCH --output=%x.%A_%a.out
#SBATCH --error=%x.%A_%a.err
#SBATCH --partition=general
#SBATCH --qos=low
#SBATCH --account=ys687
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --mem=28G

module load GCCcore/14.2.0
module load GCC/14.2.0

module load CMake/3.31.3
module load OpenMPI/5.0.7

cmake -B build -S .
cmake --build build -j --clean-first 2>/dev/null

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun ./build/GeMV_PipelineParallel 32768 32768
srun ./build/GeMV_PipelineParallel 32768 32768 > ./out/pipeline.out
