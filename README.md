# DFUSNN: Dual-Domain Fusion with UnSupervised Neural Networks for Parallel MRI Reconstruction

Date : November-23-2023

## Requirements and Dependencies
    python==3.9.16
    Pytorch==1.12.0
    numpy==1.24.4
    scipy==1.11.1


## KUSNNS
You can get the result of KUSNNS by running the following code
```bash
python main.py --config ./configs/config.yaml  
```

## How to fusion

When you abtain two results of sub-network, you can use 'fusion.py' to obtain the final reconstruction result

```bash
python fusion.py 
```

## Acknowledgement
The implementation is based on these repositories:  
https://github.com/ZhuoxuCui/K_UNN  
https://github.com/COMP6248-Reproducability-Challenge/selfsupervised-denoising  
https://github.com/MLI-lab/ConvDecoder  
https://github.com/byaman14/ZS-SSL  
