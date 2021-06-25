&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;

![diagram](https://i.imgur.com/Nb0123s.png)



# Bayesian Image Reconstruction using Deep Generative Models
### R. Marinescu, D. Moyer, P. Golland


For inquiries, please create a Github issue. We will reply as soon as we can.

For demos of our BRGM model, see these Colab Notebooks:
  * [Tensorflow](https://colab.research.google.com/drive/1G7_CGPHZVGFWIkHOAke4HFg06-tNHIZ4?usp=sharing)
  * [Pytorch Bayesian MAP](https://colab.research.google.com/drive/1xJAor6_Ky36gxIk6ICNP--NMBjSlYKil?usp=sharing)
  * [Pytorch Variational Inference](https://colab.research.google.com/drive/1dDH_bV3aYPSK1ujbPg9MBYf-7_Lbhyrw?usp=sharing)
  * <span style="color:red">NEW</span> [Try it on your own image](https://colab.research.google.com/drive/1dDH_bV3aYPSK1ujbPg9MBYf-7_Lbhyrw?usp=sharing)


## News

* **Next**: More datasets and examples
* **June 2021**: Reimplemented the method in Pytorch, and switched to StyleGAN-ADA.
* **May 2021**: Added variational inference extension for sampling multiple solutions. Updated methods section in [arXiv paper](https://arxiv.org/abs/2012.04567). Also included qualitative comparisons against Deep Image Prior in supplement.
* **Feb 2021**: Updated methods section in [arXiv paper](https://arxiv.org/abs/2012.04567). We now start from the full Bayesian formulation, and derive the loss function from the MAP estimate (in appendix), and show the graphical model. Code didn't change in this update.
* **Dec 2020**: Pre-trained models now available on [MIT Dropbox](https://www.dropbox.com/sh/0rj092juxauivzv/AABQoEfvM96u1ehzqYgQoD5Va?dl=0).
* **Nov 2020**: Uploaded article pre-print to [arXiv](https://arxiv.org/abs/2012.04567).

## Requirements

Our method, BRGM, builds on the StyleGAN-ADA Pytorch codebase, so our requirements are the same as for [StyleGAN2 Pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch):
* 64-bit Python 3.7 and PyTorch 1.7.1. See [https://pytorch.org/](https://pytorch.org/) for PyTorch install instructions.
* CUDA toolkit 11.0 or later.  Use at least version 11.1 if running on RTX 3090. If version 11 is not available, BRGM inference still works, but ignore the warnings with `python -W ignore recon.py` 
* Python libraries: `pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3`.  We use the Anaconda3 2020.11 distribution which installs most of these by default.
* 1&ndash;8 high-end NVIDIA GPUs with at least 12 GB of memory. We have done all testing and development using NVIDIA DGX-1 with 8 Tesla V100 GPUs.
* For running the inference from a pre-trained model, you need 1 GPU with at least 12GB of memory. We ran on NVIDIA Titan Xp. For training a new StyleGAN2-ADA generator, you need 1-8 GPUS.


## Installation from scratch with Anaconda


Create conda environment and install packages:

```
conda create -n brgmp python=3.7 
source activate brgmp

conda install pytorch==1.7.1 -c pytorch 
pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3 imageio scikit-image opencv-python pyro-ppl 

```


Clone this github repository:
```
git clone https://github.com/razvanmarinescu/brgm-pytorch.git 
```

## Download pre-trained StyleGAN2 & StyleGAN-ADA models

Download ffhq.pkl, xray.pkl and brains.pkl:
```
make downloadNets

```

The FFHQ model differs from NVidia's as it was trained on only 90% of FFHQ, leaving 10% for testing. 

## Image reconstruction through the Bayesian MAP estimate

We will run the model on the following images provided in this repo:

```
ls datasets/*

datasets/ffhq:
1.png  2.png  3.png  4.png  5.png

datasets/xray:
1.png  2.png  3.png  4.png  5.png

datasets/brains:
1.png  2.png  3.png  4.png
```

Run super-resolution with various super-resolution factors. We need to provide the input directory and the network. 
```
python -W ignore bayesmap_recon.py --inputdir=datasets/ffhq --outdir=recFFHQ --network=ffhq.pkl --recontype=super-resolution --superres-factor 64
	
python -W ignore bayesmap_recon.py  --inputdir=datasets/xray --outdir=recXRAY --network=xray.pkl --recontype=super-resolution --superres-factor 32

python -W ignore bayesmap_recon.py  --inputdir=datasets/brains --outdir=recBrains --network=brains.pkl --recontype=super-resolution --superres-factor 8
```


Inpainting on all three datasets using given mask files:
```
python -W ignore bayesmap_recon.py  --inputdir=datasets/ffhq --outdir=recFFHQinp --network=ffhq.pkl --recontype=inpaint --masks=masks/1024x1024

python -W ignore bayesmap_recon.py  --inputdir=datasets/xray --outdir=recXRAYinp --network=xray.pkl --recontype=inpaint --masks=masks/1024x1024

python -W ignore bayesmap_recon.py  --inputdir=datasets/brains --outdir=recBrainsInp --network=brains.pkl --recontype=inpaint --masks=masks/256x256
```

The provided masks can be found below. They need to have the same filename as the images in the input folder.

```
ls masks/*
 
masks/1024x1024:
0.png  1.png  2.png  3.png  4.png  5.png  6.png

masks/256x256:
0.png  1.png  2.png  3.png  4.png  5.png  6.png
```


## Reconstructing multiple solutions through Variational Inference

Super-resolution:
```
python -W ignore vi_recon.py --inputdir=datasets/ffhq --outdir=samFFHQ --network=ffhq.pkl --recontype=super-resolution --superres-factor=64

python -W ignore vi_recon.py --inputdir=datasets/xray --outdir=samXRAY --network=xray.pkl --recontype=super-resolution --superres-factor=32

python -W ignore vi_recon.py --inputdir=datasets/brains --outdir=samBrains --network=brains.pkl --recontype=super-resolution --superres-factor=8

```


In-painting:
```
python -W ignore vi_recon.py --inputdir=datasets/ffhq --outdir=samFFHQinp --network=ffhq.pkl --recontype=inpaint --masks=masks/1024x1024

python -W ignore vi_recon.py --inputdir=datasets/xray --outdir=samXRAYinp --network=xray.pkl --recontype=inpaint --masks=masks/1024x1024

python -W ignore vi_recon.py --inputdir=datasets/brains --outdir=samBrainsInp --network=brains.pkl --recontype=inpaint --masks=masks/256x256
```



## Training new StyleGAN2 generators

Follow the [StyleGAN2-ADA instructions](https://github.com/NVlabs/stylegan2-ada-pytorch) for how to train a new generator network. In short, you need to first create a .zip dataset, and then train the model:

```
python dataset_tool.py --source=datasets/ffhq --dest=myffhq.zip

python train.py --outdir=~/training-runs --data=myffhq.zip --gpus=8
```

If you use our model, please cite:
```
@article{marinescu2020bayesian,
  title={Bayesian Image Reconstruction using Deep Generative Models},
  author={Marinescu, Razvan V and Moyer, Daniel and Golland, Polina},
  journal={arXiv preprint arXiv:2012.04567},
  year={2020}
}
```