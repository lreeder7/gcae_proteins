# Graph-Convolutional Auto-Encoder

This repo contains the code used for experiments in "Graph-Based Representation Learning and Probabilistic Forecasting for Molecular Dynamics" by Leah Collis (part I of written dissertation for Stanford University PhD degree completion)

Model details can be found in "Graph-Based Representation Learning for Protein Dynamics and Tensor Network Methods for the High-Dimensional Kolmogorov Backward Equation" (2026)

## Usage details:
Run associated training file with given parameters:
```
python train_model.py \
    --data pentapeptide \
    --epochs 600 \
    --model GCAE \
    --val_int 50 \
    --num_workers 4 \
    --dr 0.1 \
    --lr 0.0005 \
    --num_layers 3 \
    --num_edges 10 \
    --latent_dim 12 \
    --latent_reg 0.000001 \
    --temp_smooth 0.000001 \
 
```
Evaluation code available in ```gcae_models/evaluate_gcae.ipynb```.

Code available for temporal models: TCN and Transformer.

Forecasting (following implementation in Chen, Yifan, et al. "Probabilistic forecasting with stochastic interpolants and f\" ollmer processes." arXiv preprint arXiv:2403.13724 (2024).) 
code available in ```forecasting/```

