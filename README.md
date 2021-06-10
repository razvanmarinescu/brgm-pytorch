&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;

![diagram](https://i.imgur.com/Nb0123s.png)



# Bayesian Image Reconstruction using Deep Generative Models
### R. Marinescu, D. Moyer, P. Golland


For inquiries, please create a Github issue. We will reply as soon as we can.

For a demo of our BRGM model, see the [Colab Notebook](https://colab.research.google.com/drive/1baIbopnkxzfwY_sAQXZHOg5UalWZMn2v?usp=sharing).

## News

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

## Installation from StyleGAN2 Tensorflow environment

If you already have a StyleGAN2 Tensorflow environment in Anaconda, you can clone that environment and additionally install the missing packages: 

```
# clone environment stylegan2 into brgm
conda create --name brgm --clone stylegan2
source activate brgm

# install missing packages
conda install -c menpo opencv
conda install scikit-image==0.17.2
```

## Installation from scratch with Anaconda


Create conda environment and install packages:

```
conda create -n brgmp python=3.7 
source activate brgmp

conda install pytorch==1.7.1 -c pytorch 
pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3 imageio scikit-image opencv-python
pip install pyro-ppl 

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

Super-resolution on different sets of unseen input images, with various super-resolution factors:
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


## Image reconstruction through Variational Inference

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

