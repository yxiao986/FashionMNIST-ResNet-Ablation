# Reproducing ResNet on FashionMNIST: An Ablation Study on Network Degradation

This repository contains the PyTorch implementation for a deep learning assignment investigating the network degradation problem, as originally described in the landmark paper *Deep Residual Learning for Image Recognition* (He et al., 2016).

The project trains and evaluates various Plain and Residual Networks (ResNets) on a re-split version of the FashionMNIST dataset to demonstrate how residual connections effectively resolve optimization difficulties in deep neural networks.

## Experimental Results

All models were trained from scratch using SGD for 20 epochs with a learning rate decay at epochs 10 and 15. The results perfectly replicate the degradation phenomenon: **Plain-44** suffers a severe performance drop compared to the shallower **Plain-20**, while **ResNet-44** and **ResNet-56** successfully overcome this issue and converge efficiently.

| Model Name | n | Layers | Parameter Count | Final Test Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| Plain-20 | 3 | 20 | 269,434 | 93.21% |
| Plain-44 | 7 | 44 | 658,298 | 89.93% |
| ResNet-20 | 3 | 20 | 269,434 | 94.00% |
| ResNet-44 | 7 | 44 | 658,298 | 92.60% |
| ResNet-56 | 9 | 56 | 852,730 | 93.05% |

## Installation & Setup
If using conda to set up an environment:
```bash
# 1. Create and activate a clean Conda environment
conda create -n resnet_env python=3.10 -y
conda activate resnet_env

# 2. Install PyTorch (Update the CUDA version according to your hardware)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 3. Install core dependencies from conda-forge to align C++ libraries
conda install -c conda-forge datasets pyarrow pandas numpy scipy scikit-learn tqdm pillow -y

# 4. Install wandb via pip
pip install wandb
```

## How to run
execute:

```bash
python trainer.py
```

## References
He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).