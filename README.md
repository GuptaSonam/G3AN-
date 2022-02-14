# G3AN++:Exploring Wide GANs with Complementary Feature Learning for Video Generation

Accepted for oral in ICVGIP 2021

This is an official implementation in PyTorch of G3AN++.

![image](https://user-images.githubusercontent.com/8327102/143731199-b1006352-7b43-4657-a902-9c9f14a1a7cb.png)
This is the overall Architecture. 

Architecture of G3 Module:
![image](https://user-images.githubusercontent.com/8327102/143731654-4007365f-eaf9-4268-aea8-939902dad3bd.png)

Proposed Dual branch Appearance Module with Adaptive Masking Layer:
![image](https://user-images.githubusercontent.com/8327102/143731662-55d4f95d-42c2-4488-87b1-16ae97b32b0f.png)


# Abstract

Video generation task is a challenging problem which involves the
modelling of complex real-world dynamics. Most of the existing
methods have designed deep networks to tackle high-dimensional
video data distributions. However, the utilization of wide networks
is still under explored. Inspired by the success of wide networks
in the image recognition literature, we present G3AN++, a threestream
generative adversarial network for video. The three streams
are spatial, temporal and spatio-temporal processing branches. In
pursuit of improving the quality of video generation, we make our
network wider by splitting the spatial stream into two parallel identical
branches learning complementary feature representations.We
further introduce a novel adaptive masking layer to impose the
complementary constraint. The masking layer encourages the parallel
branches to learn distinct and richer visual features. Extensive
quantitative and qualitative analysis demonstrates that our model
outperforms the existing state-of-the-art methods by a significant
margin onWeizmann Action, UvA-Nemo Smile and UCF101 Action
datasets. Additional exploration reveals that G3AN++ is capable of
disentangling the appearance and motion. We also show that the
proposed method can be easily extended to solve the hard task of
text-to-video generation.

# Summary

* Introduced a novel wider Generative Adversarial Network,
G3AN++ to learn a richer feature representation.
* Proposed a novel adaptive masking layer to facilitate the
learning of complementary features by identical branches in the network
* Still maintains the Appearance and motion disentanglement.
* Exhaustive evaluation is performed using Inception Score, FID and Precision-Recall curves.
* Generalized our method for Class Conditional Video generation and Text-to-video generation. 

# Qualitative Performance
![image](https://user-images.githubusercontent.com/8327102/143731395-4e2a9508-11d1-48a8-8404-e608d32eb02e.png)

# Dependencies
* Python == 3.5
* PyTorch == 1.4
* Cuda == 10.1
* CuDNN == 7.6.3
* tqdm == 4.62
* h5py == 2.10
* matplotlib == 3.0.0
* imageio--ffmpeg == 0.4.5
* sk-video == 1.1.10

# Dataset
Weizmann dataset can be downloaded from [here](https://www.wisdom.weizmann.ac.il/~vision/SpaceTimeActions.html) and use preprocess.py to preprocess the dataset. 
We also provide the preprocessed data in the data folder of the repository.  

# Installation
Install dependencies  mentioned above. 

To Train G3AN++ model on Weizmann Dataset, put the data under <> folder and run 

python train_G4_weizmann.py

To Evaluate 

# Training and Evaluation

# Citation 
If you find this project useful for your research, please use the following BibTeX entry.


# Acknowledgement
  
Part of the code is adapted from <G3AN>. We thank the authors for their contributions to the community. 




