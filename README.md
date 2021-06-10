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

## Download pre-trained models: FFHQ trained on 90\% of data, Xray, Brains

```
make downloadNets

```

## Image reconstruction with pre-trained StyleGAN2 generators


Super-resolution with pre-trained FFHQ generator, on a set of unseen input images (datasets/ffhq), with super-resolution factor x32. The tag argument is optional, and appends that string to the results folder: 
```
python -W ignore recon.py --inputdir=datasets/ffhq --outdir=recFFHQ --network=ffhq.pkl --recontype=super-resolution --superres-factor 16
```

Inpainting with pre-trained Xray generator (MIMIC III), using mask files from masks/1024x1024/ that match the image names exactly:
```
python recon.py recon-real-images --input=datasets/xray --tag=xray \
 --network=dropbox:xray.pkl --recontype=inpaint --masks=masks/1024x1024
```

Super-resolution on brain dataset with factor x8:
```
python recon.py recon-real-images --input=datasets/brains --tag=brains \
 --network=dropbox:brains.pkl --recontype=super-resolution --superres-factor 8
```

### Running on your images
For running on your images, pass a new folder with .png/.jpg images to --input. For inpainting, you need to pass an additional masks folder to --masks, which contains a mask file for each image in the --input folder.

## Training new StyleGAN2 generators

Follow the [StyleGAN2 instructions](https://github.com/NVlabs/stylegan2) for how to train a new generator network. In short, given a folder of images , you need to first prepare a TFRecord dataset, and then run the training code:

```
python dataset_tool.py create_from_images ~/datasets/my-custom-dataset ~/my-custom-images

python run_training.py --num-gpus=8 --data-dir=datasets --config=config-e --dataset=my-custom-dataset --mirror-augment=true
```

