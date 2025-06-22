# Latent-Variable-based-Optimization

_This work proposes a novel method for multi-zone optimization of TCLs with latent variables. A multi-task learning-based framework is formulated to learn latent variables and models representing the time-coupled relationship. Model-based and model-free algorithms are proposed to solve the latent variable-based optimization problem._

Codes for submitted Paper "Dimension-reduced Optimization of Multi-zone Thermostatically Controlled Loads".

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
To reproduce the experiments of the proposed methods and comparisons ('OptIden', 'OptSim', 'OriIden', and 'OriSim'), please run
```
cd Codes/
python Lat_MB.py
python Lat_MF.py
python Ori_MB.py
python Ori_MF.py
```
To reproduce the experiments of generating latent and original models, please run
```
cd Codes/
python Lat_model.py
python Ori_model.py
```
To reproduce the experiments of ground-truth results, please run
```
cd Codes/
python Ground_truth.py
```
Note: There is NO multi-GPU/parallelling training in our codes. 

The trained models and all figures are saved in ```Results```. Please refer to ```readme.md``` in the ```Results``` fold for more details.

## Citation
```
@ARTICLE{11037228,
  author={Cui, Xueyuan and Wang, Yi and Xu, Bolun},
  journal={IEEE Transactions on Smart Grid}, 
  title={Dimension-Reduced Optimization of Multi-Zone Thermostatically Controlled Loads}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Optimization;Buildings;Accuracy;Computational modeling;Temperature control;Dimensionality reduction;Aggregates;Multitasking;Heuristic algorithms;HVAC;Building energy system;thermostatically controlled loads;dimension reduction;auto-encoder},
  doi={10.1109/TSG.2025.3579778}}
```
