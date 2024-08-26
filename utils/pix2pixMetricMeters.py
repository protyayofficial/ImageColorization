import torch
from skimage.color import lab2rgb
import matplotlib.pyplot as plt
import time
import numpy as np
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from pytorch_msssim import ssim as calculate_ssim
import torch.nn.functional as F

class AverageMeter:
    """
    Computes and stores the average and current value of a metric.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.sum = 0.0
        self.avg = 0.0

    def update(self, val, count=1):
        self.sum += val * count
        self.count += count
        self.avg = self.sum / self.count


class Metrics:
    """
    Class to compute the Fréchet Inception Distance (FID) score, SSIM, and PSNR.
    """
    def __init__(self, device):
        self.device = device
        self.inception = inception_v3(weights='DEFAULT', transform_input=False).to(self.device)
        self.inception.eval()
        self.mu = None
        self.sigma = None
        # Add a resize transformation to ensure the correct image size
        self.resize_transform = transforms.Resize((224, 224))

    def calculate_fid(self, real_images, generated_images):
        """
        Calculate the FID score between real and generated images.
        """
        # Resize images to 224x224 and move them to the correct device
        real_images = torch.stack([self.resize_transform(img) for img in real_images]).to(self.device)
        generated_images = torch.stack([self.resize_transform(img) for img in generated_images]).to(self.device)

        # Preprocess images for Inception model
        real_images = real_images.view(-1, 3, 224, 224)
        generated_images = generated_images.view(-1, 3, 224, 224)

        with torch.no_grad():
            real_features = self.inception(real_images)
            generated_features = self.inception(generated_images)

        real_features = real_features.view(real_features.size(0), -1)
        generated_features = generated_features.view(generated_features.size(0), -1)

        if self.mu is None:
            self.mu = real_features.mean(dim=0)
            self.sigma = real_features.std(dim=0)

        real_mu = real_features.mean(dim=0)
        real_sigma = real_features.std(dim=0)

        generated_mu = generated_features.mean(dim=0)
        generated_sigma = generated_features.std(dim=0)

        mu_diff = real_mu - generated_mu
        sigma_diff = real_sigma - generated_sigma

        fid = mu_diff.pow(2).sum() + (self.sigma - generated_sigma).pow(2).sum() + (self.mu - generated_mu).pow(2).sum()
        return fid.item()

    def calculate_ssim(self, real_images, generated_images):
        """
        Calculate SSIM (Structural Similarity Index) between real and generated images.
        """
        # Ensure that real and generated images are of the same shape
        real_images = torch.stack(real_images).to(self.device)
        generated_images = torch.stack(generated_images).to(self.device)

        # Convert images to a 4D tensor (batch_size, channels, height, width)
        real_images = real_images.unsqueeze(0) if real_images.dim() == 3 else real_images
        generated_images = generated_images.unsqueeze(0) if generated_images.dim() == 3 else generated_images

        # Compute SSIM
        return calculate_ssim(generated_images, real_images, data_range=1.0, size_average=True).item()

    def calculate_psnr(self, real_images, generated_images):
        """
        Calculate PSNR (Peak Signal-to-Noise Ratio) between real and generated images.
        """
        # Ensure that real and generated images are of the same shape
        real_images = torch.stack(real_images).to(self.device)
        generated_images = torch.stack(generated_images).to(self.device)

        mse = F.mse_loss(generated_images, real_images)
        if mse == 0:
            return float('inf')
        
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr.item()
    

def create_loss_meters():
    """
    Create a dictionary of AverageMeter objects for each loss type.
    """
    return {
        'discriminator_loss_fake': AverageMeter(),
        'discriminator_loss_real': AverageMeter(),
        'discriminator_loss': AverageMeter(),
        'generator_loss_GAN': AverageMeter(),
        'generator_loss_L1': AverageMeter(),
        'generator_loss': AverageMeter(),
    }


def update_losses(model, loss_meter_dict, count):
    """
    Update the loss meters with the current losses from the model.
    """
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)


def lab_to_rgb(L, ab):
    """
    Convert LAB color space tensors to RGB.
    """
    L = (L + 1.) * 50.  # Rescale L channel to [0, 100]
    ab = ab * 110.  # Rescale ab channels to [-110, 110]

    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()  # Combine and permute dimensions
    rgb_imgs = [lab2rgb(img) for img in Lab]  # Convert LAB to RGB

    return np.stack(rgb_imgs, axis=0)


def visualize(model, data, SAVE_DIR, epoch, save=True):
    """
    Visualize the grayscale input, predicted colorization, and ground truth colorization.
    """
    model.generator_model.eval()

    with torch.no_grad():
        model.setup_input(data)
        model.forward()

    model.generator_model.train()

    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L

    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)

    fig = plt.figure(figsize=(15, 8))
    for i in range(5):
        # Grayscale Input (L channel)
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis('off')

        # Predicted Colorization
        ax = plt.subplot(3, 5, i + 6)
        ax.imshow(fake_imgs[i])
        ax.axis('off')

        # Ground Truth Colorization
        ax = plt.subplot(3, 5, i + 11)
        ax.imshow(real_imgs[i])
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig(f"{SAVE_DIR}/pix2pix_{epoch}.png")


def log_results(loss_meter_dict):
    """
    Log the average losses in a clean, formatted manner.
    """
    print("\n===== Loss Metrics =====")
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name.replace('_', ' ').title()}: {loss_meter.avg:.5f}")
    print("========================\n")


