# hdc2021_StepNet (In Progress)
One of the submissions from the ZeTeM Uni Bremen Team for the HDC2021. 

## Install 
Install the package using:

```
pip install -e .
```
Make sure to have git-lfs installed to pull the weight files for the model. 

If the bandwith limit of the git is reached, you can also download the weights from this link: https://seafile.zfn.uni-bremen.de/d/0d90fed4f14b45bf9213/

## Usage 
Prediction on images in a folder can be done using:

```
python hdc2021_challenge/main.py path-to-input-files path-to-output-files step
```

## Method
The StepNet $F: Y \rightarrow X$ is a fully-learned and purely data-driven inversion model. It directly maps blurry measurements $y^\delta$ to reconstructions $\hat{x}$. The StepNet itself consists of 20 sub-networks $F_i$, ($i=0,...,19$), which are connected in sequence $F_0 \circ F_1 \circ ... \circ F_{19}$. The task of a sub-network $F_i$ is to receive an input with blurring level $i$ and produce an output at blurring level $i-1$. For our implementation of the StepNet model, we use 20 small U-Nets for the sub-networks.

### Reconstruction
The structure of the network is dynamic and directly depends on the current blurring step. Let's consider a blurry image $y^\delta_i$ at step $i$. The first $i+1$ sub-networks will be active for the reconstruction process: 

$$\hat{x} = F_0 \circ F_1 \circ ... \circ F_i(y^\delta_i)$$.

Notice: The model will use more GPU memory and take longer for higher blurring steps.

### Training
The StepNet training involves 20 different steps to gradually adapt the parameters of each sub-network $F_i$. During the training at step $i$, only the parameters of $F_i$ can be changed. All other weights are frozen. We start the training at blurring step $i=0$ and run this training for a fixed number of epochs. Afterwards, the best parameter combination w.r.t. the OCR performance on the validation set is selected and used for the specific sub-network. This produce is repeated until the end of step $i=19$ is reached.

Since the StepNet is a purely daten-driven approach, there is a high probability that it will perform poorly on out-of-distribution data. Therefore, to enhance the robustness of the model, we also use simulated blurry samples from the STL10 dataset during training.

## Requirements 
* numpy = 1.20.3
* pytorch = 1.9.0 
* pytorch-lightning = 1.3.8
* torchvision = 0.10.0
* dival = 0.6.1

## Authors
Team University of Bremen, Center of Industrial Mathematics (ZeTeM) et al.: 
- Alexander Denker
- Maximilian Schmidt
- Johannes Leuschner
- Sören Dittmer
- Judith Nickel
- Clemens Arndt
- Gael Rigaud
- Richard Schmähl
