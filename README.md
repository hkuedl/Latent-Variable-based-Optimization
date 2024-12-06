# Latent-Variable-based-Optimization

_This work proposes a novel method for multi-zone optimization of TCLs with latent variables. A multi-task learning based framework is formulated to learn latent variables and models that represent the time-coupled relationship. Both model-based and model-free algorithms are proposed to solve the latent variable-based optimization problem._

Codes for submitted Paper "XXXXX".

Authors: Xueyuan Cui, Yi Wang, and Bolun Xu.

## Requirements
Python version: 3.8.17

The must-have packages can be installed by running
```
pip install requirements.txt
```

## Experiments
### Data
All the data for experiments can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1U4RE0EGJgCrL_LJvFmMf_LiXID7o4P38?usp=sharing).

### Reproduction
To reproduce the experiments of the proposed method, please run
```
cd Codes/
python Archive_Model_ODE_OPT_Grad.py
```
To reproduce the experiments of generating initial models and comparisons, please run
```
cd Codes/
python Archive_Model_ODE_then_OPT.py
```
Note: There is NO multi-GPU/parallelling training in our codes. 

The models and logs are saved in ```Results/Archive_NNfile/```, and the case results are saved in ```Results/Archive_Results/```.

Please refer to ```readme.md``` in each fold for more details.

## Citation
```
```
