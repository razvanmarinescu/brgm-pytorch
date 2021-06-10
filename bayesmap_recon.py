# Author: Razvan Marinescu
# razvan@csail.mit.edu
# Adapted from NVIDIA's Stylegan-ADA projector script

"""Image reconstruction through the Bayesian MAP estimate using pre-trained StyleGAN-ADA."""

import copy
import os
from time import perf_counter

import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import skimage.io

import dnnlib
import legacy

from forwardModels import *


def cosine_distance(latentsBLD):
  # assert latentsBLD.shape[0] == 1
  cosDist = 0
  for b in range(latentsBLD.shape[0]):
    latentsNormLD = F.normalize(latentsBLD[0, :, :], dim=1, p=2)
    cosDistLL = 1 - torch.matmul(latentsNormLD, latentsNormLD.T)
    cosDist += cosDistLL.reshape(-1).norm(p=1)
  return cosDist


def constructForwardModel(recontype, imgSize, nrChannels, mask_dir, imgShort, superres_factor, image_idx, device):
  if recontype == 'none':
    forward = ForwardNone();
    forwardTrue = forward  # no forward model, just image inversion

  elif recontype == 'super-resolution':
    # Create downsampling forward corruption model
    forward = ForwardDownsample(factor=superres_factor);
    forwardTrue = forward  # res = target resolution

  elif recontype == 'inpaint':
    # Create forward model that fills in part of image with zeros (change the height/width to control the bounding box)
    forward = ForwardFillMask(device)
    maskFile = '%s/%s' % (mask_dir, imgShort)  # mask should have same name as image
    print('Loading mask %s' % maskFile)
    mask = skimage.io.imread(maskFile)
    mask = mask[:, :, 0] == np.min(mask[:, :, 0])  # need to block black color

    mask = np.reshape(mask, (1, 1, mask.shape[0], mask.shape[1]))
    forward.mask = torch.tensor(np.repeat(mask, nrChannels, axis=1), dtype=torch.bool, device=device)
    forwardTrue = forward
  else:
    raise ValueError('recontype has to be either none, super-resolution, inpaint')

  return forward, forwardTrue


def getVggFeatures(images, num_channels, vgg16):
  # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
  # if synth_images_down.shape[2] > 256:
  images_vgg = F.interpolate(images, size=(256, 256), mode='area')

  if num_channels == 1:
    # if grayscale, move back to RGB to evaluate perceptual loss
    images_vgg = images_vgg.repeat(1, 3, 1, 1)  # BCWH

  # Features for synth images.
  features = vgg16(images_vgg, resize_images=False, return_lpips=True)
  return features


def project(
    G,
    forward,  # forward corruption model (downsampling, masking, ...). Can even be identity, so no corruption
    target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    num_steps,
    filepath,
    w_avg_samples=10000,
    verbose=False,
    device: torch.device,
    recontype,
    lambda_pix,
    lambda_perc,
    lambda_w,
    lambda_c,
    save_progress
):
  def logprint(*args):
    if verbose:
      print(*args)

  G = copy.deepcopy(G).eval().requires_grad_(False).to(device)  # type: ignore

  # Compute w stats.
  logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
  z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
  w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
  w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
  w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
  w_std = torch.tensor(np.std(w_samples, axis=0, keepdims=True), dtype=torch.float32, device=device)
  w_std_scalar = torch.tensor((np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5, dtype=torch.float32,
                              device=device)
  w_avg = torch.tensor(w_avg, dtype=torch.float32, device=device)  # [1, 1, C]

  # Setup noise inputs.
  noise_bufs = {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}

  # Load VGG16 feature detector.
  url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
  with dnnlib.util.open_url(url) as f:
    vgg16 = torch.jit.load(f).eval().to(device)

  # Features for target image.
  target_images = target.to(device).to(torch.float32)
  target_features = getVggFeatures(target_images, G.img_channels, vgg16)

  ws = torch.tensor(w_avg.repeat([1, G.mapping.num_ws, 1]), dtype=torch.float32, device=device, requires_grad=True)
  ws_out = torch.zeros([num_steps] + list(ws.shape[1:]), dtype=torch.float32, device=device)

  noiseLayers = list(noise_bufs.values())

  optimizerAdam = torch.optim.Adam([ws],
                                   betas=(0.9, 0.999),
                                   lr=0.1)

  # Init noise.
  for buf in noiseLayers:
    buf[:] = torch.randn_like(buf)  # changed zeros_like to rand_like
    buf.requires_grad = False

  for step in range(num_steps):

    def closure():

      optimizerAdam.zero_grad()

      # Synth images from opt_w.
      synth_images = G.synthesis(ws, noise_mode='const')  # G(w)

      # renormalise back to 0-255
      synth_images = (synth_images + 1) * (255 / 2)

      ######### FORWARD #########
      # apply forward model
      synth_images_down = forward(synth_images)  # f(G(w))
      ####### END-FORWARD ##########

      assert target_images.shape[1:] == synth_images_down.shape[1:]

      # adding L2 loss in pixel space
      pixelwise_loss = lambda_pix * (synth_images_down - target_images).square().mean()
      loss = 0
      loss += pixelwise_loss

      # perceptual loss
      synth_features = getVggFeatures(synth_images_down, G.img_channels, vgg16)
      perceptual_loss = lambda_perc * (target_features - synth_features).square().mean()
      loss += perceptual_loss

      # adding prior on w ~ N(mu, sigma) as extra loss term
      w_loss = lambda_w * (
            ws / w_std_scalar - w_avg / w_std_scalar).square().mean()  # will broadcast w_avg: [1, 1, 512] to ws: [1, L, 512]
      loss += w_loss

      # adding cosine distance loss
      cosine_loss = lambda_c * cosine_distance(ws)
      loss += cosine_loss

      loss.backward(create_graph=False)

      return loss, pixelwise_loss, perceptual_loss, w_loss, cosine_loss, synth_images, synth_images_down

    loss, pixelwise_loss, perceptual_loss, w_loss, cosine_loss, synth_images, synth_images_down = optimizerAdam.step(closure=closure)
    logprint(f'step {step + 1:>4d}/{num_steps}: tloss {float(loss):<5.4f} pix_loss {float(pixelwise_loss):<5.2f} '
             f'perc_loss {float(perceptual_loss):<5.2f} w_loss {float(w_loss):<5.2f} cos_loss {float(cosine_loss):<5.2f}  ')

    # save progress so far
    if save_progress and step % 100 == 0:
      saveImage(image=synth_images[0, :, :, :], filepath='%s_clean_step%d.jpg' % (filepath, step))
      saveImage(image=synth_images_down[0, :, :, :], filepath='%s_corrupted_step%d.jpg' % (filepath, step),
                target_res=(G.img_resolution, G.img_resolution))

      if recontype.startswith('inpaint'):
        merged = torch.where(forward.mask, synth_images[0, :, :, :], target_images)  # if true, then synth, else target
        saveImage(image=merged[0, :, :, :], filepath='%s_merged_step%d.jpg' % (filepath, step))



    # Save projected W for each optimization step.
    ws_out[step] = ws.detach()[0]

  return ws_out


def saveImage(image, filepath, target_res=None):
  ''' image = CHW (no batch dimension anymore)'''
  # print('image.shape', image.shape)
  chan = image.shape[0]
  image = image.permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy().squeeze()
  # print('image.shape', image.shape)
  if chan == 3:
    pilimg = PIL.Image.fromarray(image, 'RGB')
  else:
    # assume grayscale
    pilimg = PIL.Image.fromarray(image, 'L')

  if target_res is not None:
    pilimg = pilimg.resize(target_res, PIL.Image.NEAREST)

  pilimg.save(filepath)


# ----------------------------------------------------------------------------


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--inputdir', 'inputdir', help='Folder of input (target) images to project to', required=True,
              metavar='FILE')
@click.option('--num-steps', help='Number of optimization steps', type=int, default=501, show_default=True)
@click.option('--seed', help='Random seed', type=int, default=303, show_default=True)
@click.option('--save-video', help='Save an mp4 video of optimization progress', type=bool, default=True,
              show_default=False)
@click.option('--outdir', help='Where to save the output images', required=True, metavar='DIR')
@click.option('--recontype',
              help='Type of reconstruction: "none" (normal image inversion), "super-resolution", "inpaint" ',
              required=True)
@click.option('--superres-factor', help='Super-resolution factor: 2,4,8,16,32,64,128,256 ', type=int, default=32)
@click.option('--masks',
              help='Directory with masks (inpainting only). Mask filenames should be identical to the input filenames.',
              default='masks')
@click.option('--save-progress', help='Save optimisation progress as jpg images, every X steps.', default=False)
def run_projection(
    network_pkl: str,
    inputdir: str,
    outdir: str,
    save_video: bool,
    seed: int,
    num_steps: int,
    recontype: str,
    superres_factor: int,
    masks: str,
    save_progress: bool,
):
  """Project given image to the latent space of pretrained network pickle.

  Examples:

  \b
  python projector.py --outdir=out --target=~/mytargetimg.png \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
  """
  np.random.seed(seed)
  torch.manual_seed(seed)

  # Load networks.
  print('Loading networks from "%s"...' % network_pkl)
  device = torch.device('cuda')
  with dnnlib.util.open_url(network_pkl) as fp:
    G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)  # type: ignore

  os.makedirs(outdir, exist_ok=True)

  imageList = np.sort(list(os.listdir(inputdir)))
  imageList = [f for f in imageList if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")]

  if len(imageList) == 0:
    raise ValueError('folder does not contain any images')

  lambda_pix = 0.001
  lambda_perc = 10000000
  lambda_w = 100
  lambda_c = 0.1  # tried many values, this is a good one for in-painting

  if recontype == 'super-resolution' and superres_factor > 16:
    lambda_c = 1

  # Load target images.
  image_idx = 0
  for filename in imageList:

    fullpath = os.path.join(inputdir, filename)
    fnshort = filename.split('.')[0]
    print('loading ', fullpath)

    true_pil = PIL.Image.open(fullpath)
    print('target PIL shape', true_pil.size)
    chan = G.img_channels  # number of channels
    if chan == 3:
      true_pil = true_pil.convert('RGB')
    else:
      true_pil = true_pil.convert('L')
    print('target PIL shape', true_pil.size)

    w, h = true_pil.size
    s = min(w, h)
    true_pil = true_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    true_pil = true_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    true_pil.save(f'{outdir}/{fnshort}_true.jpg')
    true_float = np.array(true_pil, dtype=np.float).reshape((G.img_resolution, G.img_resolution, -1))

    forward, _ = constructForwardModel(recontype, G.img_resolution, G.img_channels, masks, filename,
                                       1 / superres_factor, image_idx, device)
    image_idx += 1  # for rotating through inpainting masks

    true_tensor_float = torch.tensor(true_float.transpose([2, 0, 1]), device=device)
    print('true_tensor_float.shape', true_tensor_float.shape)
    target = forward(true_tensor_float.unsqueeze_(0))  # pass through forward model to generate corrupted image
    saveImage(target[0], filepath=f'{outdir}/{fnshort}_target.jpg',
              target_res=(G.img_resolution, G.img_resolution))

    # Optimize projection.
    start_time = perf_counter()
    filepath = f'{outdir}/{fnshort}'
    ws_out = project(
      G,
      forward=forward,
      target=target,  # pylint: disable=not-callable
      num_steps=num_steps,
      device=device,
      verbose=True,
      filepath=filepath,
      recontype=recontype,
      lambda_pix=lambda_pix,
      lambda_perc=lambda_perc,
      lambda_w=lambda_w,
      lambda_c=lambda_c,
      save_progress=save_progress
    )
    print(f'Elapsed: {(perf_counter() - start_time):.1f} s')

    # Render debug output: optional video and projected image and W vector.
    if save_video:
      video = imageio.get_writer(f'{outdir}/{fnshort}_proj.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
      print(f'Saving optimization progress video "{outdir}/{fnshort}_proj.mp4"')
      for ws in ws_out:
        synth_image = G.synthesis(ws.unsqueeze(0), noise_mode='const')
        synth_image = (synth_image + 1) * (255 / 2)
        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        video.append_data(np.concatenate([true_float.astype(np.uint8), synth_image], axis=1))
      video.close()

    print(f'Saving final images: mean_clean, mean_corrupted and the merged image (inpainting only)')
    # First re-synthesize the images from the optimised latents ws_out, and also pass them through the corruption model
    ws = ws_out[-1].unsqueeze(0)
    synth_images = G.synthesis(ws, noise_mode='const')
    synth_images = (synth_images + 1) * (255 / 2)
    synth_images_corrupted_mean = forward(synth_images)  # f(G(w))

    # save the mean image (i.e. sample 0 with zero noise)
    saveImage(image=synth_images[0, :, :, :], filepath='%s_clean.jpg' % filepath)
    saveImage(image=synth_images_corrupted_mean[0, :, :, :], filepath='%s_corrupted.jpg' % filepath,
              target_res=(G.img_resolution, G.img_resolution))

    # save merged images for inpainting
    target_images = target.to(device).to(torch.float32)
    if recontype.startswith('inpaint'):
      merged = torch.where(forward.mask, synth_images[0, :, :, :],
                           target_images)  # if true, then synth, else target
      saveImage(image=merged[0, :, :, :], filepath='%s_merged.jpg' % filepath)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
  run_projection()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
