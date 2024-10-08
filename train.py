import os
import time
import torch
import numpy as np
import random
import argparse
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.pix2pixInitWeights import init_model
from utils.pix2pixLoss import GANLoss
from models.pix2pix import MainModel
from utils.pix2pixDataLoader import make_dataloaders
from utils.pix2pixMetricMeters import create_loss_meters, update_losses, visualize, log_results, Metrics, lab_to_rgb, AverageMeter
from tqdm import tqdm
from torchvision.models.resnet import resnet18
import torch.optim as optim

from fastai.vision.learner import create_body
from fastai.vision.models.unet import DynamicUnet

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def set_seed(seed):
    """
    Set seed for reproducibility.

    Parameters:
        seed (int): Seed value to set.

    Returns:
        None
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_epoch(model, data_loader, loss_meters):
    """
    Train the model for one epoch.

    Parameters:
        model (MainModel): The model to be trained.
        data_loader (DataLoader): The data loader for training data.
        loss_meters (dict): Dictionary of loss meters.

    Returns:
        None
    """
    model.train()
    for data in tqdm(data_loader):
        model.setup_input(data)
        model.optimize()
        
        # Update losses
        update_losses(model, loss_meters, count=data['L'].size(0))

def save_model(model, best_fid, current_fid, current_ssim, current_psnr, save_dir, model_name):
    """
    Save the best model based on the lowest FID score and log the metrics.

    Parameters:
        model (MainModel): The model to be saved.
        best_fid (float): The best FID score so far.
        current_fid (float): The current FID score.
        current_ssim (float): The current SSIM score.
        current_psnr (float): The current PSNR score.
        save_dir (str): Directory where models will be saved.
        model_name (str): Base name for saving the model files.

    Returns:
        float: The updated best FID score.
    """
    # Save the best model if the current FID is lower than the best FID
    if current_fid < best_fid:
        best_fid = current_fid
        save_path = os.path.join(save_dir, f"best_model_{model_name}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"[INFO] Saved best model with FID: {current_fid:.4f} at {save_path}")

    # Log the current FID, SSIM, and PSNR values
    print(f"[INFO] Current Metrics:")
    print(f"  FID   : {current_fid:.4f}")
    print(f"  SSIM  : {current_ssim:.4f}")
    print(f"  PSNR  : {current_psnr:.4f} dB")
    
    return best_fid


def process_batch(batch, model):
    """
    Process a batch of images through the model.

    Parameters:
        batch (dict): A dictionary containing input data with keys 'L' and 'ab'. 
                      'L' represents the lightness channel, and 'ab' represents the color channels.
        model (MainModel): The Pix2Pix model used for image colorization.

    Returns:
        tuple: A tuple containing two lists of tensors:
            - real_images (list of torch.Tensor): The ground truth color images converted to tensors.
            - generated_images (list of torch.Tensor): The colorized images generated by the model, converted to tensors.
    """
    # Setup the model with the input batch
    model.setup_input(batch)  # Replace with your model's input setup method
    
    # Perform a forward pass through the model
    model.forward()  # Forward pass
    
    # Extract the generated and real color images
    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L
    
    # Convert the LAB images to RGB color space
    generated_images = lab_to_rgb(L, fake_color)
    real_images = lab_to_rgb(L, real_color)
    
    # Convert the images to PyTorch tensors
    generated_images = [transforms.ToTensor()(img) for img in generated_images]
    real_images = [transforms.ToTensor()(img) for img in real_images]
    
    return real_images, generated_images

def build_backbone_unet(in_channels=1, out_channels=2, img_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body = create_body(resnet18(), pretrained=True, n_in=in_channels, cut=-2)
    generator = DynamicUnet(encoder=body, n_out=out_channels, img_size=(img_size, img_size)).to(device)

    return generator

def pretrain_generator(generator, train_dataloader, optimizer, criterion, epochs, device):
    for e in range(epochs):
        loss_meter = AverageMeter()
        for data in tqdm(train_dataloader):
            L, ab = data['L'].to(device), data['ab'].to(device)
            preds = generator(L)
            loss = criterion(preds, ab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), L.size(0))

        print(f"[INFO] Epoch: {e + 1} / {epochs}")
        print(f"L1 Loss: {loss_meter.avg:.5f}")


def main():
    """
    Main training loop for the Pix2Pix model.

    Returns:
        None
    """

    parser = argparse.ArgumentParser(description="Train Pix2Pix model for image colorization.")
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and validation.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for data loading.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training.')
    parser.add_argument('--visualize_interval', type=int, default=10, help='Interval for visualization.')
    parser.add_argument('--metrics_interval', type=int, default=1, help='Interval for metrics calculation.')
    parser.add_argument('--seed', type=int, default=9, help='Seed for reproducibility.')
    parser.add_argument('--save_dir', type=str, default='experiments', help='Directory to save models and logs.')
    parser.add_argument('--model_name', type=str, default='pix2pix', help='Base name for saving model files.')
    parser.add_argument('--train_data_path', type=str, default='coco/test2017', help='Path to training data.')
    parser.add_argument('--val_data_path', type=str, default='coco/val2017', help='Path to validation data.')

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Print hyperparameters
    print(f"[INFO] Hyperparameters:")
    print(f"  Batch Size         : {args.batch_size}")
    print(f"  Number of Workers  : {args.num_workers}")
    print(f"  Number of Epochs   : {args.num_epochs}")
    print(f"  Visualization Interval : {args.visualize_interval}")
    print(f"  Metrics Interval   : {args.metrics_interval}")
    print(f"  Seed               : {args.seed}")
    print(f"  Save Directory     : {args.save_dir}")
    print(f"  Model Name         : {args.model_name}")
    print(f"  Training Data Path : {args.train_data_path}")
    print(f"  Validation Data Path: {args.val_data_path}")

    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)

    # Paths
    # TRAIN_PATHS = np.array([os.path.join('coco/train2017', fname) for fname in os.listdir('coco/train2017')])
    VAL_PATHS = np.array([os.path.join(args.val_data_path, fname) for fname in os.listdir(args.val_data_path)])
    TRAIN_PATHS = np.array([os.path.join(args.train_data_path, fname) for fname in os.listdir(args.train_data_path)])


    # Data loaders
    train_loader = make_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, paths=TRAIN_PATHS, split='train')
    print(f"[INFO] Training Data loaded. Found {len(train_loader.dataset)} images for training.")
    val_loader = make_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, paths=VAL_PATHS, split='val')
    print(f"[INFO] Validation Data loaded. Found {len(val_loader.dataset)} images for validation.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = build_backbone_unet(in_channels=1, out_channels=2, img_size=256)
    optimizer = optim.AdamW(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    loss = nn.L1Loss()
    pretrain_generator(generator=generator, train_dataloader=train_loader, optimizer=optimizer, criterion=loss, epochs=args.num_epochs, device=device)
    
    torch.save(generator.state_dict(), "resnet18-unet.pth")

    generator.load_state_dict(torch.load("resnet18-unet.pth", map_location=device))

    # Initialize model
    model = MainModel(generator_model=generator, generator_lr=2e-4, discriminator_lr=2e-4, beta1=0.5, beta2=0.999, lambda_L1=100.)
    model = init_model(model, device=device)

    # Loss meters
    loss_meters = create_loss_meters()

    # Initialize FID
    metrics_calculator = Metrics(device)

    best_fid = float('inf')


    fids = []
    ssims = []
    psnrs = []

    # Training loop
    print(f"[INFO] Training starting ...")
    for epoch in range(args.num_epochs):
        start_time = time.time()

        # Train for one epoch
        train_epoch(model, train_loader, loss_meters)

        # Log results
        print(f"[INFO] Epoch [{epoch + 1}/{args.num_epochs}] - Time: {time.time() - start_time:.2f}s")
        log_results(loss_meters)

        # Visualize results
        if (epoch) % args.visualize_interval == 0:
            visualize(model, next(iter(val_loader)), args.save_dir, epoch, save=True)

        if (epoch) % args.metrics_interval == 0:
            fids = []
            ssims = []
            psnrs = []
            with torch.no_grad():
                for batch in tqdm(val_loader):
                    real_images, generated_images = process_batch(batch, model)
                    fid_score = metrics_calculator.calculate_fid(real_images, generated_images)
                    ssim_score = metrics_calculator.calculate_ssim(real_images, generated_images)
                    psnr_score = metrics_calculator.calculate_psnr(real_images, generated_images)
                    fids.append(fid_score)
                    ssims.append(ssim_score)
                    psnrs.append(psnr_score)

            # Calculate average metrics for this epoch
            avg_fid = np.mean(fids)
            avg_ssim = np.mean(ssims)
            avg_psnr = np.mean(psnrs)

            print(f"\n[INFO] Epoch [{epoch + 1}/{args.num_epochs}] Results:")
            print(f"{'-'*50}")
            print(f"  FID Score   : {avg_fid:.4f}")
            print(f"  Mean SSIM   : {avg_ssim:.4f}")
            print(f"  Mean PSNR   : {avg_psnr:.4f} dB")
            print(f"{'-'*50}\n")

            # Save best model based on FID score
            best_fid = save_model(model, best_fid, avg_fid, avg_ssim, avg_psnr, args.save_dir, args.model_name)


        # Reset loss meters
        for meter in loss_meters.values():
            meter.reset()

    print("Training complete.")

if __name__ == "__main__":
    main()
