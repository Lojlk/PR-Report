#!/bin/bash
#SBATCH --partition=comp3710 
#SBATCH --account=comp3710 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100
#SBATCH --job-name=2_ViT_train
#SBATCH -o Res_2_ViT.out

conda activate demo

# Run the training script
python train_2.py

# Run the prediction script after training
python predict_2.py
