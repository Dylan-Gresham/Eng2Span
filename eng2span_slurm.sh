#!/bin/bash
#SBATCH -J eng2span		# job name
#SBATCH -o log_slrm.o%j		# output and error file name (%j expands to jobID)
#SBATCH -n 1			# total number of tasks requested
#SBATCH -c 48			# cpus cores per task
#SBATCH -N 1			# number of nodes to run on
#SBATCH --gres=gpu:L40:1	# request gpu L40 is specific, 1 is number of
#SBATCH -p gpu-l40		# queue (partition)
#SBATCH -t 96:00:00		# runtime (DD-hh:mm:ss) 96:00:00 max
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT	# I love emails, but can edit
#SBATCH --mail-user=emmagifford@u.boisestate.edu	# swap for you, unless both?

# (optional) debug info!
echo "Date		= $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated  = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated  = $SLURM_NTASKS"
echo "GPU Allocated              = $SLURM_JOB_GPUS"
echo ""
nvidia-smi

# activate engironment
. ~/.bashrc
conda activate py39 # replace with a joint conda environment
                    # this one is just python 3.9 from conda tutorial

# load module
module load cudnn8.0-cuda11.0/8.0.5.39 # to use GPU cuda stuff


# run script
python pretraining.py # replace with script

