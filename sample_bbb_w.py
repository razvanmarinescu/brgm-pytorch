# Author: Razvan Marinescu
# razvan@csail.mit.edu
# Based on NVIDIA Stylegan-ADA projector script

"""Project given image to the latent space of pretrained network pickle."""

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
from pyro.distributions.inverse_gamma import InverseGamma

def cosine_distance(latentsBLD):
  # assert latentsBLD.shape[0] == 1
  cosDist = 0
  for b in range(latentsBLD.shape[0]):
    latentsNormLD = F.normalize(latentsBLD[0,:,:], dim=1, p=2)
    cosDistLL = 1 - torch.matmul(latentsNormLD, latentsNormLD.T)
    cosDist += cosDistLL.reshape(-1).norm(p=1)
  return cosDist

def constructForwardModel(recontype, imgSize, nrChannels, mask_dir, imgShort, superres_factor, image_idx, device):
  if recontype == 'none':
    forward = ForwardNone(); forwardTrue = forward # no forward model, just image inversion

  elif recontype == 'super-resolution':
    # Create downsampling forward corruption model
    forward = ForwardDownsample(factor=superres_factor); forwardTrue = forward # res = target resolution

  elif recontype == 'inpaint':
    # Create forward model that fills in part of image with zeros (change the height/width to control the bounding box)
    forward = ForwardFillMask(device)
    maskFile = '%s/%s' % (mask_dir, imgShort) # mask should have same name as image 
    print('Loading mask %s' % maskFile)
    mask = skimage.io.imread(maskFile)
    mask = mask[:,:,0] == np.min(mask[:,:,0]) # need to block black color

    mask = np.reshape(mask, (1,1, mask.shape[0], mask.shape[1]))
    forward.mask = torch.tensor(np.repeat(mask, nrChannels, axis=1), dtype=torch.bool, device=device)
    forwardTrue = forward
  elif recontype == 'inpaint-eval':
    # Create forward model that fills in part of image with zeros (change the height/width to control the bounding box)
    forward = ForwardFillMask(device)
    maskInd = image_idx % 7
    if imgSize == 1024:
        mask = skimage.io.imread('masks_eval/%d.png' % maskInd)
    else:
        mask = skimage.io.imread('masks_eval_brains/%d.png' % maskInd)
    mask = mask[:,:,0] == np.min(mask[:,:,0]) # need to block black

    mask = np.reshape(mask, (1,1, mask.shape[0], mask.shape[1]))
    forward.mask = torch.tensor(np.repeat(mask, nrChannels, axis=1), dtype=torch.bool, device=device)
    forwardTrue = forward
    #print(forward.mask.shape)
    #asda

  else:
    raise ValueError('recontype has to be either none, super-resolution, inpaint')

  return forward, forwardTrue

class ConstModule(torch.nn.Linear):
  def __init__(self, ws): # defined as 0*x + ws = ws. Only bias is optimised, the weights are set to seros.
    in_features = 0
    out_features = np.max(list(ws.shape))
    super(ConstModule, self).__init__( in_features, out_features, bias = True)
    self.weight = torch.nn.Parameter(torch.zeros(out_features, in_features), requires_grad=False)
    self.bias = torch.nn.Parameter(ws)

  def reset_parameters(self) -> None:
    pass

def project(
    G,
    forward, # forward corruption model (downsampling, masking, ...). Can even be identity, so no corruption
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    num_steps,
    filepath,
    w_avg_samples              = 10000,
    initial_learning_rate      = 0.1,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    regularize_noise_weight    = 1000,
    verbose                    = False,
    device: torch.device,
    recontype,
    wsInit = None,
    lambda_pix,
    lambda_perc,
    lambda_w,
    lambda_c
):

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    w_std = torch.tensor(np.std(w_samples, axis=0, keepdims=True) , dtype=torch.float32, device=device)
    w_std_scalar = torch.tensor((np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5, dtype=torch.float32,  device=device)
    w_avg = torch.tensor(w_avg, dtype=torch.float32, device=device)      # [1, 1, C]

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.to(device).to(torch.float32)
    #if target_images.shape[2] > 256:
    target_images_vgg = F.interpolate(target_images, size=(256, 256), mode='area') #[BCWH]

    print('target_images.shape', target_images.shape)
    if G.img_channels == 1:
      # if grayscale, move back to RGB to evaluate perceptual loss
      target_images_vgg = target_images_vgg.repeat(1,3,1,1) # BCWH
    print('target_images.shape', target_images.shape)

    target_features = vgg16(target_images_vgg, resize_images=False, return_lpips=True) # vgg16 takes BHW only



    if wsInit is not None:
      w_mu = torch.tensor(wsMuInit, device=device, requires_grad=True)
      # w_std = torch.tensor(wsStdInit.repeat([1, G.mapping.num_ws, 1]))
      w_rho = None
    else:
      w_mu = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True)
      w_rho = torch.tensor(torch.log(torch.exp(w_std) - 1), dtype=torch.float32, device=device, requires_grad=True)

    print('w_mu.shape',w_mu.shape)
    print('w_std.shape', w_std.shape)
    print('w_rho.shape', w_rho.shape)

    # w_out = torch.zeros([num_steps] + list(w_mu.shape[1:]), dtype=torch.float32, device=device)
    w_out = None

    noiseLayers = list(noise_bufs.values())
    # maxRes = 1024
    maxRes = 0
    noiseLayersToOpt = [l for l in noiseLayers if l.shape[0] <= maxRes]  # each layer is 2D: WxH
    noiseLayersFixed = [l for l in noiseLayers if l.shape[0] > maxRes]


    optimizerAdam = torch.optim.Adam([w_mu, w_rho] + noiseLayersToOpt,
                     betas=(0.9, 0.999),
                     lr=0.1)

    concentration = torch.tensor(0.1, device=device) # alpha
    rate = torch.tensor(0.95, device=device) # beta
    # mode is b/(a+1)    mean = b(a-1) for a > 1


    prior_sigma = InverseGamma(concentration=concentration, rate=rate)

    # Init noise.
    # for buf in noise_bufs.values():
    for buf in noiseLayersToOpt:
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for buf in noiseLayersFixed:
        buf[:] = torch.randn_like(buf) # changed zeros_like to rand_like
        buf.requires_grad = False

    nrS = 6


    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std_scalar * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        #lr = initial_learning_rate
        #for param_group in optimizer.param_groups:
        #    param_group['lr'] = lr


        def closure():

          eps = torch.randn((nrS - 1, w_rho.shape[1], w_rho.shape[2]), dtype=torch.float32, device=device)
          zeroNoise = torch.zeros((1, w_rho.shape[1], w_rho.shape[2]), dtype=torch.float32, device=device)
          eps = torch.cat((zeroNoise, eps), dim=0)

          optimizerAdam.zero_grad()

          # Synth images from opt_w.
          #w_noise = torch.randn_like(w_opt) * w_noise_scale
          #w_noise = torch.randn_like(w_mu) * w_noise_scale

          w_std_deriv = torch.log(1 + torch.exp(w_rho))

          w = eps * w_std_deriv + w_mu

          ws = w.repeat([1, G.mapping.num_ws, 1])


          # print('w.shape', w.shape)

          synth_images = G.synthesis(ws, noise_mode='const') #G(w)
          #synth_images = G.synthesis(w_mu + w_noise, noise_mode='const') #G(w)

          # renormalise back to 0-255
          synth_images = (synth_images + 1) * (255/2)

          ######### FORWARD #########
          # apply forward model
          synth_images_down = forward(synth_images) # f(G(w))
          ####### END-FORWARD ##########


          # print('synth_images_down.shape', synth_images_down.shape) # BCHW
          # print('target_images.shape', target_images.shape) # 1CHW
          #print('synth_images_down', synth_images_down)
          #print('target_images', target_images)
          #print('target_images min-max', torch.min(target_images), torch.max(target_images))
          #print('synth_images_down min-max', torch.min(synth_images_down), torch.max(synth_images_down))
          assert target_images.shape[1:] == synth_images_down.shape[1:]

          # adding L2 loss in pixel space
          #self._l2_pixelwise_reg = tf.placeholder(tf.float32, [], name='lambda_pix')
          #self._l2_pixelwise_reg = torch.tensor(0.001, dtype=tf.float32)

          pixelwise_loss = lambda_pix * (synth_images_down - target_images).square().mean()
          loss = 0
          loss += pixelwise_loss


          # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
          #if synth_images_down.shape[2] > 256:
          synth_images_down_vgg = F.interpolate(synth_images_down, size=(256, 256), mode='area')

          if G.img_channels == 1:
            # if grayscale, move back to RGB to evaluate perceptual loss
            synth_images_down_vgg = synth_images_down_vgg.repeat(1,3,1,1) # BCWH

          # Features for synth images.
          synth_features = vgg16(synth_images_down_vgg, resize_images=False, return_lpips=True)
          # print('synth_features.shape', synth_features.shape)
          # print('target_features.shape', target_features.shape)

          perceptual_loss = lambda_perc * (target_features - synth_features).square().mean()
          loss += perceptual_loss


          # adding prior on w ~ N(mu, sigma) as extra loss term
          #lambda_w = 10

          w_loss = lambda_w * (w/w_std_scalar - w_avg/w_std_scalar).square().mean() # will broadcast w_avg: [1, 1, 512] to w_mu: [1, L, 512]
          #w_loss = 0
          loss += w_loss

          # adding cosine distance loss
          #lambda_c = 0.001 # lambda_c
          #lambda_c = 0.1 # lambda_c

          #lambda_c = 0.0
          cosine_loss = lambda_c * cosine_distance(w_mu)
          #cosine_loss = 0
          loss += cosine_loss

          # Noise regularization.
          reg_loss = 0
          for v in noise_bufs.values():
             noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
             while True:
                 reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                 reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                 if noise.shape[2] <= 8:
                     break
                 noise = F.avg_pool2d(noise, kernel_size=2)
          noise_loss = reg_loss * regularize_noise_weight
          noise_loss = 0
          #loss += noise_loss


          # add variational posterior loss
          # log q(w|theta)
          q_loss = -0.5 * ((w - w_mu) / w_std_deriv).square().mean()

          # prior on mu/variance
          # print('w_std_deriv min', torch.min(w_std_deriv))
          # print('w_std_deriv max', torch.max(w_std_deriv))

          assert torch.min(w_std_deriv) > 0.0
          # print('example log_prob', prior_sigma.log_prob(torch.tensor(torch.min(w_std_deriv), device=device)))
          # print('example log_prob', prior_sigma.log_prob(torch.tensor(torch.max(w_std_deriv), device=device)))
          # print('log prob', prior_sigma.log_prob(torch.tensor([0.1, 0.5, 0.86, 1.0, 1.5, 2.0],device=device)))
          theta_loss = - prior_sigma.log_prob(w_std_deriv/2).mean() # negative sign, to maximise the log_prob
          #theta_loss = 0
          # print('w_std_deriv', w_std_deriv[0,:,0])
          # print('theta_loss',theta_loss.shape)
          loss += q_loss + theta_loss

          loss.backward(create_graph=False)

          return loss, pixelwise_loss, perceptual_loss, w_loss, cosine_loss, noise_loss, theta_loss, synth_images, synth_images_down
          #return loss



        # Example closure function to use as reference
        # def closure():
          # optimizer.zero_grad()
          # synth_images = G.synthesis(w_mu, noise_mode='const')  # G(w)
          # output = model(data)
          # loss = F.cross_entropy(output, target)
          # loss.backward(create_graph=args.create_graph)
          # return loss, output

        # Step
        #print(closure())
        #asdsa


        loss, pixelwise_loss, perceptual_loss, w_loss, cosine_loss, noise_loss, theta_loss, synth_images, synth_images_down = optimizerAdam.step(closure=closure)
        logprint(f'step {step+1:>4d}/{num_steps}: tloss {float(loss):<5.4f} pix_loss {float(pixelwise_loss):<5.2f} '
                 f'perc_loss {float(perceptual_loss):<5.2f} w_loss {float(w_loss):<5.2f} cos_loss {float(cosine_loss):<5.2f}  '
                 f'noise_loss {float(noise_loss):<5.3f} theta_loss {float(theta_loss):<5.3f}')

        # if loss > 2000:
        #   ads
        #   break #  the sampler can sometimes breakdown

        #loss = optimizer.step(closure=closure)

        #outputs = optimizer.get_mean_predictions(model.forward, inputs=x_train, mc_samples=eval_mc_samples,
        #                                           ret_numpy=False)



        if step > -1 and step % 10 == 0:
         saveImage(image=synth_images[0,:,:,:], filepath='%s_clean_step%d.jpg' % (filepath, step))
         saveImage(image=synth_images_down[0,:,:,:], filepath='%s_corrupted_step%d.jpg' % (filepath, step),
                   target_res=(G.img_resolution, G.img_resolution))
         for s in range(1,nrS):
           saveImage(image=synth_images[s, :, :, :], filepath='%s_sample%d_step%d.jpg' % (filepath, s, step))
           saveImage(image=synth_images_down[s, :, :, :], filepath='%s_corrsample%d_step%d.jpg' % (filepath, s, step),
                     target_res=(G.img_resolution, G.img_resolution))

         if recontype.startswith('inpaint'):
           merged = torch.where(forward.mask, synth_images[0,:,:,:], target_images)  # if true, then synth, else target
           saveImage(image=merged[0,:,:,:], filepath='%s_merged_step%d.jpg' % (filepath, step))

           for s in range(1, nrS):
             merged = torch.where(forward.mask, synth_images[s, :, :, :], target_images)  # if true, then synth, else target
             saveImage(image=merged[0, :, :, :], filepath='%s_mergedsample%d_step%d.jpg' % (filepath, s, step))

        # Save projected W for each optimization step.
        # w_out[step] = w_mu.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noiseLayersToOpt:
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()


    return w_out

def saveImage(image, filepath, target_res=None):
    ''' image = CHW (no batch dimension anymore)'''
    #print('image.shape', image.shape)
    chan = image.shape[0]
    image = image.permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy().squeeze()
    #print('image.shape', image.shape)
    if chan == 3:
        pilimg = PIL.Image.fromarray(image, 'RGB')
    else:
        #assume grayscale
        pilimg = PIL.Image.fromarray(image, 'L')

    if target_res is not None:
      pilimg = pilimg.resize(target_res, PIL.Image.NEAREST)

    pilimg.save(filepath)

#----------------------------------------------------------------------------




@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--inputdir', 'inputdir',         help='Folder of input (target) images to project to', required=True, metavar='FILE')
@click.option('--num-steps',              help='Number of optimization steps', type=int, default=500, show_default=True)
@click.option('--seed',                   help='Random seed', type=int, default=303, show_default=True)
@click.option('--save-video',             help='Save an mp4 video of optimization progress', type=bool, default=False, show_default=False)
@click.option('--outdir',                 help='Where to save the output images', required=True, metavar='DIR')
@click.option('--recontype',              help='Type of reconstruction: "none" (normal image inversion), "super-resolution", "inpaint" ', required=True)
@click.option('--superres-factor',              help='Super-resolution factor: 2,4,8,16,32,64 ', type=int, default=4)
@click.option('--masks',              help='Directory with masks (inpainting only). Mask filenames should be identical to the input filenames.', default='masks')
def run_projection(
    network_pkl: str,
    inputdir: str,
    outdir: str,
    save_video: bool,
    seed: int,
    num_steps: int,
    recontype: str,
    superres_factor:int,
    masks: str
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
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    imageList = list(os.listdir(inputdir))
    imageList = [f for f in imageList if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")]

    if len(imageList) == 0:
      raise ValueError('folder does not contain any images')


    lambda_pix = 0.001
    lambda_perc = 10000000
    lambda_w = 100
    lambda_c = 0.1  # tried many values, this is a good one for in-painting

    if recontype == 'super-resolution' and superres_factor > 16:
      lambda_w = 100
      lambda_c = 1




    # Load target images.
    image_idx = 0
    for filename in imageList:

        fullpath = os.path.join(inputdir, filename)
        fnshort = filename.split('.')[0]
        print('loading ', fullpath)


        true_pil = PIL.Image.open(fullpath)
        print('target PIL shape', true_pil.size)
        chan = G.img_channels # number of channels
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
        #true_uint8 = np.array(true_pil, dtype=np.uint8).reshape((G.img_resolution, G.img_resolution, -1))
        true_uint8 = np.array(true_pil, dtype=np.float).reshape((G.img_resolution, G.img_resolution, -1))
        
        forward, _ = constructForwardModel(recontype, G.img_resolution, G.img_channels, masks, filename, 1/superres_factor, image_idx, device)
        image_idx += 1 # for rotating through inpainting masks

        if image_idx == 1:
          continue

        #true_tensor_uint8 = torch.tensor(true_uint8.transpose([2,0,1])[np.newaxis,:], device=device)
        true_tensor_uint8 = torch.tensor(true_uint8.transpose([2,0,1]), device=device)
        print('true_tensor_uint8.shape', true_tensor_uint8.shape)
        target_uint8 = forward(true_tensor_uint8.unsqueeze_(0)) # pass through forward model to generate corrupted image
        #target_uint8.save(f'{outdir}/{fnshort}_target.jpg')
        saveImage(target_uint8[0], filepath=f'{outdir}/{fnshort}_target.jpg', target_res=(G.img_resolution,G.img_resolution))

        # wsInit = np.load(f'out/{fnshort}_projected_w.npz')['w']

        # Optimize projection.
        start_time = perf_counter()
        projected_w_steps = project(
            G,
            forward=forward,
            target=target_uint8, # pylint: disable=not-callable
            num_steps=num_steps,
            device=device,
            verbose=True,
            filepath=f'{outdir}/{fnshort}',
            recontype=recontype,
            wsInit=None,
            lambda_pix= lambda_pix,
            lambda_perc=lambda_perc,
            lambda_w=lambda_w,
            lambda_c=lambda_c
        )
        print (f'Elapsed: {(perf_counter()-start_time):.1f} s')

        # Render debug output: optional video and projected image and W vector.

        # projected_w = projected_w_steps[-1]
        # synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
        # synth_image = (synth_image + 1) * (255/2)
        # saveImage(synth_image[0], filepath=f'{outdir}/{fnshort}_proj.jpg')


        # if save_video:
        #     video = imageio.get_writer(f'{outdir}/{fnshort}_proj.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
        #     print (f'Saving optimization progress video "{outdir}/{fnshort}_proj.mp4"')
        #     for projected_w in projected_w_steps:
        #         synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
        #         synth_image = (synth_image + 1) * (255/2)
        #         synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        #         video.append_data(np.concatenate([true_uint8, synth_image], axis=1))
        #     video.close()

        # Save final projected frame and W vector.



        #np.savez(f'{outdir}/{fnshort}_projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())


#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_projection() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
