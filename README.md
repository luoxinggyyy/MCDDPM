# MCDDPM
_Article title: A multi-condition diffusion model controls the reconstruction of 3D digital rocks_  
_Journal title: Computers and Geosciences_
## Description
![MCDDPM](https://github.com/luoxinggyyy/MCDDPM/blob/main/22.png)
##  Usage
python+pytorch  
GPU: RTX 3060 or 3060+

---
## Installation
   pip install -r `requirement.txt`
   
---
## Test
   We provide pre-trained models for heterogeneous carbonate rocks, and if you want to try to generate digital cores, you can run `MainCondition_new`.py file while the pre-trained model is placed in the `CheckpointsCondition` folder.  
The generated result will be saved in the `npydata` folder.

---
## Train
  Please run `MainCondition_new.py` and change the 'state' to 'train' and 'path' to the location of your dataset in `MainCondition_new.py`.
### Train description
   Regarding the training parameters in `MainCondition_new.py`, 'epoch' represents the number of training iterations, 'batch_size' refers to the number of training batches, and 'T' represents the time step in the diffusion model equation, typically set to 1000. A value of 500 will result in lower resolution effects. 'channel' represents the number of channels to adjust based on hardware requirements. 'label' corresponds to the porosity parameter, 'labelA' represents the average pore diameter, and 'labelB' represents the standard deviation of pore diameter. For a description of other parameters, please refer to the paper.
   
---
## Acknowledgements
   Although we have proposed a relatively new approach, the initial idea and design were inspired by (https://video-diffusion.github.io/), and the code structure was inspired by （https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/tree/main）
   
---
## License
   meanderpy is licensed under the Apache License 2.0.
   
---


