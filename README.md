# HDC 2021 StepNet
One of the submissions from the ZeTeM Uni Bremen Team for the Helsinki Deblur Challenge 2021 (HDC 2021).
https://www.fips.fi/HDC2021.php

Team members are listed below.

## Requirements
The main requirements for our code are listed below. You can also use the requirements.txt file to replicate our conda environment.
* numpy = 1.20.3
* pytorch = 1.9.0
* pytorch-lightning = 1.3.8
* torchvision = 0.10.0
* dival = 0.6.1
* torchvision = 0.10.0
* pytesseract = 0.3.8
* fuzzywuzzy = 0.18.0

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
The StepNet <a href="https://www.codecogs.com/eqnedit.php?latex=F:&space;Y&space;\rightarrow&space;X" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F:&space;Y&space;\rightarrow&space;X" title="F: Y \rightarrow X" /></a> is a fully-learned and purely data-driven inversion model. It directly maps blurry measurements <a href="https://www.codecogs.com/eqnedit.php?latex=y^\delta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y^\delta" title="y^\delta" /></a> to reconstructions <a href="https://www.codecogs.com/eqnedit.php?latex=\hat{x}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{x}" title="\hat{x}" /></a>. The StepNet itself consists of 20 sub-networks <a href="https://www.codecogs.com/eqnedit.php?latex=F_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F_i" title="F_i" /></a>, <a href="https://www.codecogs.com/eqnedit.php?latex=(i=0,...,19)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(i=0,...,19)" title="(i=0,...,19)" /></a>, which are connected in sequence <a href="https://www.codecogs.com/eqnedit.php?latex=F_0&space;\circ&space;F_1&space;\circ&space;...&space;\circ&space;F_{19}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F_0&space;\circ&space;F_1&space;\circ&space;...&space;\circ&space;F_{19}" title="F_0 \circ F_1 \circ ... \circ F_{19}" /></a>. The task of a sub-network <a href="https://www.codecogs.com/eqnedit.php?latex=F_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F_i" title="F_i" /></a> is to receive an input with blurring level <a href="https://www.codecogs.com/eqnedit.php?latex=i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?i" title="i" /></a> and produce an output at blurring level <a href="https://www.codecogs.com/eqnedit.php?latex=i-1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?i-1" title="i-1" /></a>. For our implementation of the StepNet model, we use 20 small U-Nets for the sub-networks.

### Reconstruction
The structure of the network is dynamic and directly depends on the current blurring step. Let's consider a blurry image <a href="https://www.codecogs.com/eqnedit.php?latex=y^\delta_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y^\delta_i" title="y^\delta_i" /></a> at step <a href="https://www.codecogs.com/eqnedit.php?latex=i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?i" title="i" /></a>. The first <a href="https://www.codecogs.com/eqnedit.php?latex=i&plus;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?i&plus;1" title="i+1" /></a> sub-networks will be active for the reconstruction process:

<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{x}&space;=&space;F_0&space;\circ&space;F_1&space;\circ&space;...&space;\circ&space;F_i(y^\delta_i)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{x}&space;=&space;F_0&space;\circ&space;F_1&space;\circ&space;...&space;\circ&space;F_i(y^\delta_i)" title="\hat{x} = F_0 \circ F_1 \circ ... \circ F_i(y^\delta_i)" /></a>.

Notice: The model will use more GPU memory and take longer for higher blurring steps.

### Training
The StepNet training involves 20 different steps to gradually adapt the parameters of each sub-network <a href="https://www.codecogs.com/eqnedit.php?latex=F_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F_i" title="F_i" /></a>. During the training at step <a href="https://www.codecogs.com/eqnedit.php?latex=i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?i" title="i" /></a>, only the parameters of <a href="https://www.codecogs.com/eqnedit.php?latex=F_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F_i" title="F_i" /></a> can be changed. All other weights are frozen. We start the training at blurring step <a href="https://www.codecogs.com/eqnedit.php?latex=i=0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?i=0" title="i=0" /></a> and run this training for a fixed number of epochs. Afterwards, the best parameter combination w.r.t. the OCR performance on the validation set is selected and used for the specific sub-network. This produce is repeated until the end of step <a href="https://www.codecogs.com/eqnedit.php?latex=i=19" target="_blank"><img src="https://latex.codecogs.com/gif.latex?i=19" title="i=19" /></a> is reached.

Since the StepNet is a purely daten-driven approach, there is a high probability that it will perform poorly on out-of-distribution data. Therefore, to enhance the robustness of the model, we also use simulated blurry samples from the STL10 dataset during training.

### Reference results
OCR accuracy on our test set (20 images per step):
- 0: 90.85
- 1: 91.55
- 2: 93.35
- 3: 95.35
- 4: 94.10
- 5: 88.45
- 6: 85.50
- 7: 92.95
- 8: 86.80
- 9: 85.70
- 10: 83.65
- 11: 79.65
- 12: 75.70
- 13: 67.40
- 14: 68.35
- 15: 57.25
- 16: 38.70
- 17: 25.05
- 18: 23.15
- 19: 17.10

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

## Examples
Random reconstructions from the test set on different blur steps:

### Step 2
![Blur step 2](example_images/step_2test_sample0.png "Step 2")

### Step 5
![Blur step 5](example_images/step_5test_sample9.png "Step 5")

### Step 10
![Blur step 10](example_images/step_10test_sample5.png "Step 10")

### Step 12
![Blur step 12](example_images/step_12test_sample3.png "Step 12")

### Step 15
![Blur step 15](example_images/step_15test_sample17.png "Step 15")

### Step 19
![Blur step 19](example_images/step_19test_sample18.png "Step 19")
