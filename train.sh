#!/usr/bin/env bash

# Ron the short-list GPU queue
#SBATCH --partition=slurm_sbel_cmg
#SBATCH --account=skunkworks --qos=skunkworks_owner

## Request one CPU core from the scheduler
#SBATCH -c 1

## Request a GPU from the scheduler, we don't care what kind
##SBATCH --gres=gpu:gtx2080ti:1
#SBATCH -t 4-16:00 # time (D-HH:MM)

## Create a unique output file for the job
#SBATCH --error=/srv/home/nkrakauer/chemnetEuler/silaneEngAmodel-%j.err
#SBATCH --output=/srv/home/nkrakauer/chemnetEuler/silaneEngAmodel-%j.out

## Load CUDA into your environment
## load custimized CUDA and cudaToolkit

module load usermods
module load user/cuda

# activate retina virtual environment
source activate moleprop

# install tensorflow and other libraries for machine learning

#conda install -c conda-forge keras --name moleprop
#conda install -c rdkit --name moleprop rdkit

#export HOME="/srv/home/nkrakauer/"
#export CUDA_HOME=/usr/local/cuda
#export PATH=$PATH:$CUDA_HOME/bin
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64:$HOME/extras/CUPTI/lib64

#python /srv/home/nkrakauer/chemnetEuler/ToxNet_01_CleanData.py
#python /srv/home/nkrakauer/chemnetEuler/ToxNet_01_CleanExtData.py
#python /srv/home/nkrakauer/chemnetEuler/ToxNet_02_Internal_Test_Set.py
#python /srv/home/nkrakauer/chemnetEuler/ToxNet_03_SplitTask_Int.py
#python /srv/home/nkrakauer/chemnetEuler/ToxNet_03_SplitTask_TrainVal.py
#python /srv/home/nkrakauer/chemnetEuler/ToxNet_21_Prep2D.py
#python /srv/home/nkrakauer/chemnetEuler/ToxNet_31_PrepSMILES.py
python /srv/home/nkrakauer/chemnetEuler/MLP_Prototype.py
