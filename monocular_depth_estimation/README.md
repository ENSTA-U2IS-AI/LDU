# LDU on BTS
Latent Discriminant deterministic Uncertainty   
[Paper](https://arxiv.org/abs/2207.10130)

From Big to Small: Multi-Scale Local Planar Guidance for Monocular Depth Estimation   
[Paper](https://arxiv.org/abs/1907.10326)  
[Supplementary material](https://arxiv.org/src/1907.10326v4/anc/bts_sm.pdf) 

## Note
This folder contains a PyTorch implementation of BTS with the plugged LDU solution.\
We tested this code under python 3.8, PyTorch 1.11.0, CUDA 10.2 on Ubuntu 20.04.\
Our experiment is applied only on KITTI dataset but we think it is also feasible on NYU dataset.\
We modified as well the procedure of evaluation.

## Modifications
`pytorch/bts_ldu.py`: based on `pytorch/bts.py`, the extra loss functions are added, as well as the plugged LDU module.\
`pytorch/bts_test_kitti_ldu.py`: based on `pytorch/bts_test.py` and `utils/eval_with_pngs.py`, the two-step evaluation procedure is simplified, now the evaluation is online.\
`pytorch/arguments_test_eigen_ldu`: arguments for evaluation.
## Preparation for Training

### KITTI
You can train BTS with KITTI dataset by following procedures.
First, make sure that you have prepared the ground truth depthmaps from [KITTI](http://www.cvlibs.net/download.php?file=data_depth_annotated.zip).
If you have not, please follow instructions on README.md at root of this repo.
Then, download and unzip the raw dataset using following commands.
```
$ cd ~/workspace/dataset/kitti_dataset
$ aria2c -x 16 -i ../../bts_ldu/utils/kitti_archives_to_download.txt
$ parallel unzip ::: *.zip
```
Then we need to modify the `arguments_test_eigen_ldu.txt`. Finally, we can train our network with
```
$ cd ~/workspace/bts_ldu/pytorch
$ python bts_main.py arguments_train_eigen.txt
```


### NYU Depvh V2
Download the dataset we used in this work.
```
$ cd ~/workspace/bts
$ python utils/download_from_gdrive.py 1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP ../dataset/nyu_depth_v2/sync.zip
$ unzip sync.zip
```
Also, you can download it from following link:
https://drive.google.com/file/d/1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP/view?usp=sharing
Please make sure to locate the downloaded file to ~/workspace/bts/dataset/nyu_depth_v2/sync.zip

Once the dataset is ready, you can train the network using following command.
```
$ cd ~/workspace/bts_ldu/pytorch
$ python bts_main.py arguments_train_nyu.txt
```
You can check the training using tensorboard:
```
$ tensorboard --logdir ./models/bts_nyu_test/ --port 6006
```
Open localhost:6006 with your favorite browser to see the progress of training.

## Evaluation

### Testing and Evaluation with KITTI
Once you have KITTI dataset and official ground truth depthmaps, we need to modify `arguments_test_eigen_ldu.txt`, then we can evaluate the prediction results with following command.
```
$ cd ~/workspace/bts/pytorch
$ python bts_test_kitti_ldu.py arguments_test_eigen_ldu.txt
```

## Acknowledgements
The code is largely builds on the [BTS](https://github.com/cleinc/bts) repository. Thanks the authors of [BTS](https://arxiv.org/abs/1907.10326).   


## License
The original BTS implementation:\
Copyright (C) 2019 Jin Han Lee, Myung-Kyu Han, Dong Wook Ko and Il Hong Suh \
This Software is licensed under GPL-3.0-or-later.


According to the GPL-3.0-or-later license, we release our codes.\
This modified version is also licensed uncer GPL-3.0-or-later.