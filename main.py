import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion,Trainer
from multiprocessing import freeze_support
from torchvision import transforms as T, utils
import warnings

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def load(model,milestone):

    data = torch.load(str(f'./results/model-{milestone}.pt'), map_location='cuda')
    model.load_state_dict(data['model'])

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = False,
    self_condition = False,
    learned_sinusoidal_cond = False
)

diffusion = GaussianDiffusion(
    model,
    image_size = 32,
    timesteps = 50,   # number of steps
    use_prior = True,
    T = 0.999,
    eps = 0.0001,
    is_ddim = True,
    mode = 'gaussian',
    consistent = True
)
#sampled_images.shape # (4, 3, 128, 128)

trainer = Trainer(
    diffusion,
    r"C:\Users\Zonglin\Desktop\CIFAR-10-images-master\train",
    train_batch_size = 2,
    train_lr = 1e-5,
    train_num_steps = 1000000,         # total training steps
    gradient_accumulate_every = 60,    # gradient accumulation steps
    ema_decay = 0.9999,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = True,              # whether to calculate fid during training
    save_and_sample_every = 5000
)
warnings.filterwarnings("ignore")
if __name__ == '__main__':
    train = True
    mile_stone = 29
    if train:
        freeze_support()
        #trainer.load(mile_stone)
        trainer.train()
    else:
        load(diffusion,mile_stone)
        diffusion.eval()
        with torch.inference_mode():

            all_images = diffusion.sample(batch_size=1)
        utils.save_image(all_images, str(f'sample-test.png'))

##all_images = torch.cat(all_images_list, dim=0)

##utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'))