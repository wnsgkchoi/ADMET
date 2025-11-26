#!/bin/bash

# Kill the old process if the user agrees (commented out for safety, user can run manually)
# kill -9 1585105

# Create results directory if it doesn't exist
mkdir -p workspace/benchmark/results

# Run 5 seeds in parallel across 4 GPUs
# We will schedule:
# Seed 1 -> GPU 0
# Seed 2 -> GPU 1
# Seed 3 -> GPU 2
# Seed 4 -> GPU 3
# Seed 5 -> GPU 0 (after Seed 1 finishes)

echo "Starting Seed 1 on GPU 0..."
nohup python -u workspace/benchmark/baseline/run_baseline.py --dataset AMES --n_trials 10 --num_seeds 5 --seed_index 0 --gpu_id 0 > workspace/benchmark/results/log_seed_1.txt 2>&1 &

echo "Starting Seed 2 on GPU 1..."
nohup python -u workspace/benchmark/baseline/run_baseline.py --dataset AMES --n_trials 10 --num_seeds 5 --seed_index 1 --gpu_id 1 > workspace/benchmark/results/log_seed_2.txt 2>&1 &

echo "Starting Seed 3 on GPU 2..."
nohup python -u workspace/benchmark/baseline/run_baseline.py --dataset AMES --n_trials 10 --num_seeds 5 --seed_index 2 --gpu_id 2 > workspace/benchmark/results/log_seed_3.txt 2>&1 &

echo "Starting Seed 4 on GPU 3..."
nohup python -u workspace/benchmark/baseline/run_baseline.py --dataset AMES --n_trials 10 --num_seeds 5 --seed_index 3 --gpu_id 3 > workspace/benchmark/results/log_seed_4.txt 2>&1 &

# Wait for Seed 1 to finish before starting Seed 5 on GPU 0
# (Simple way: just sleep or wait? Better to just launch it and let the user manage, 
# or use a simple wait loop. For now, let's just print the command for Seed 5)

echo "Seeds 1-4 launched in background."
echo "To run Seed 5, wait for one of the GPUs to free up (e.g., GPU 0) and run:"
echo "nohup python -u workspace/benchmark/baseline/run_baseline.py --dataset AMES --n_trials 10 --num_seeds 5 --seed_index 4 --gpu_id 0 > workspace/benchmark/results/log_seed_5.txt 2>&1 &"
