# Numerically Stable Coded Matrix Multiplication

This repository contains Python implementations and experimental codes for an optimization-based framework for numerically stable, straggler-resilient coded matrix multiplication. The proposed approach optimizes encoding coefficients to improve numerical stability while maintaining accurate matrix product recovery from a subset of worker nodes.

## Repository Structure

The repository is organized into three folders:

### 1. [Different Optimizers and Initializers](Different%20Optimizers%20and%20Initializers/)

Contains Python codes for optimizing encoding and decoding coefficients using different optimization algorithms and initialization strategies. These experiments investigate their effects on convergence, recovery error, and the condition numbers of decoding matrices.

### 2. [Sparse_Masking](Sparse_Masking/)

Contains Python codes for introducing sparsity into the encoding coefficients. The sparsity masks are applied during optimization to restrict the number of nonzero encoding coefficients while evaluating matrix recovery accuracy and numerical stability.

### 3. [AWS EC2 Experiments](AWS%20EC2%20Experiments/)

Contains Python codes for distributed matrix multiplication experiments conducted on Amazon EC2 instances using MPI. These experiments evaluate worker computation time, communication time, and transmission size under different numerical precision settings. They also support comparisons between the proposed approach and conventional coded matrix multiplication methods.

## Research Objective

The main objective of this work is to develop numerically stable encoding strategies for distributed matrix multiplication while considering straggler resilience, recovery accuracy, sparsity, and practical computation and communication costs.
