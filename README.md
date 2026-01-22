# Service Application for Recommender Systems with Sparse Autoencoders

This project is a research-oriented experimental platform for designing, executing, and analyzing recommender system pipelines based on Sparse Autoencoders (SAE). It is developed within the research initiative at Charles University.

## Purpose
The goal of this project is to provide a modular, reproducible, and extensible framework that simplifies experimentation with SAE-enhanced recommender systems, with a focus on interpretability, evaluation, and steering of recommendations.

## Key Features
- Plugin-based pipeline architecture for flexible experiment composition
- Support for multi-step pipelines, including:
  - data loading and preprocessing
  - training collaborative filtering autoencoders
  - training embedded sparse autoencoders
  - neuron labeling and labeling evaluation
  - inspection and steering of recommendations
- Experiment tracking and reproducibility via MLflow
- Web-based UI for pipeline creation, execution, and result inspection
- Reuse of intermediate results from previous experiments

## Context
The platform generalizes prior SAE-based recommender research to enable systematic comparison of methods and efficient collaboration within the research group.
