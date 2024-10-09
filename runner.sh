#!/bin/bash
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100
#SBATCH --job-name=ViT_train_predict
#SBATCH -o Result_ViT.out
#SBATCH -e Error_ViT.err
#SBATCH --time=1-00:00:00 #1 days total runtime, enable to use GPUs with max times
conda activate demo

# Run the training script
python train.py

# Run the prediction script after training
python predict.py
