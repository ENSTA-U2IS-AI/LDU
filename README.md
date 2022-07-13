# LDU: Latent Discriminant deterministic Uncertainty
Latent Discriminant deterministic Uncertainty (ECCV 2022)   
[Paper]()   
[Supplementary material]()

## Abstract
In this work we advance a scalable and effective Deterministic Uncertainty Methods (DUM) that relaxes the Lipschitz constraint typically hindering practicality of such architectures. We learn a discriminant latent space by leveraging a distinction maximization layer over an arbitrarily-sized set of trainable prototypes. 

Overview of LDU:
the DNN learns a discriminative latent space thanks to
learnable prototypes. The DNN backbone computes a feature vector z for an input x and then the DM layer matches it with the prototypes. The computed similarities reflecting the position of z in the learned feature space, are subsequently processed by the classification layer and the uncertainty estimation layer. The dashed arrows point to the loss functions that need to be optimized for training LDU.

![image](https://github.com/ENSTA-U2IS/LDU/blob/main/process.png)

For more details, please refer to our paper.

## Note
We currently only provide the codes for toy example, classification and monocular depth estimation.\
The semantic segmentation part will be released in near future.

## Toy example
We provide a toy example for illustrating LDU on two-moon dataset.
<p>
<a href="https://colab.research.google.com/drive/10On0ubqVEcOUvKTNCED1_qF9l5UG7OSc?usp=sharing" target="_parent">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
</p>

## Monocular depth estimation example
In folder `monocular_depth_estimation/`, we provide the codes and instructions for LDU applying on monocular depth estimation task. The detailed information is shown on `monocular_depth_estimation/README.md`.

## TODO
-   Add classification codes


## Acknowledgements
If you find this work useful for your research, please consider citing our paper:
```
@article{

}
```
