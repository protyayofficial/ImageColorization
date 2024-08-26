import os
import time
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils.pix2pixInitWeights import init_model
from utils.pix2pixLoss import GANLoss
from models.pix2pix import  MainModel
from utils.pix2pixDataLoader import make_dataloaders
from utils.pix2pixMetricMeters import create_loss_meters, update_losses, visualize, log_results, Metrics, lab_to_rgb
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


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

def save_model(model, best_loss, current_loss, save_dir, model_name):
    """
    Save the best model and checkpoint models.

    Parameters:
        model (MainModel): The model to be saved.
        epoch (int): The current epoch.
        best_loss (float): The best loss value so far.
        current_loss (float): The current loss value.
        save_dir (str): Directory where models will be saved.
        model_name (str): Base name for saving the model files.

    Returns:
        float: The updated best loss value.
    """
    if current_loss < best_loss:
        best_loss = current_loss
        torch.save(model.state_dict(), os.path.join(save_dir, f"best_model_{model_name}.pth"))
        print(f"[INFO] Saved best model at {save_dir}/best_model_{model_name}.pth")


    return best_loss

def process_batch(batch, model):
    model.setup_input(batch)  # Replace with your model's input setup method
    model.forward()  # Forward pass
    
    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L
    
    # Convert the LAB images to RGB
    generated_images = lab_to_rgb(L, fake_color)
    real_images = lab_to_rgb(L, real_color)
    
    # Convert images to tensors
    generated_images = [transforms.ToTensor()(img) for img in generated_images]
    real_images = [transforms.ToTensor()(img) for img in real_images]
    
    return real_images, generated_images

def main():
    """
    Main training loop for the Pix2Pix model.

    Returns:
        None
    """

    BATCH_SIZE = 32
    N_WORKERS = 4
    PIN_MEMORY = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_EPOCHS = 100
    VISUALIZE_INTERVAL = 10
    METRICS_INTERVAL = 10
    SAVE_DIR = "experiments"
    MODEL_NAME = "pix2pix"

    # Create directories
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Paths
    # TRAIN_PATHS = np.array([os.path.join('coco/train2017', fname) for fname in os.listdir('coco/train2017')])
    VAL_PATHS = np.array([os.path.join('coco/val2017', fname) for fname in os.listdir('coco/val2017')])
    TRAIN_PATHS = np.array([os.path.join('coco/test2017', fname) for fname in os.listdir('coco/test2017')])


    # Data loaders
    train_loader = make_dataloaders(batch_size=BATCH_SIZE, num_workers=N_WORKERS, pin_memory=PIN_MEMORY, paths=TRAIN_PATHS, split='train')

    print(f"[INFO] Training Data loaded. Found {len(train_loader.dataset)} images for training.")

    val_loader = make_dataloaders(batch_size=BATCH_SIZE, num_workers=N_WORKERS, pin_memory=PIN_MEMORY, paths=VAL_PATHS, split='val')

    print(f"[INFO] Validation Data loaded. Found {len(val_loader.dataset)} images for validation.")
    

    # Initialize model
    model = MainModel(generator_lr=2e-4, discriminator_lr=2e-4, beta1=0.5, beta2=0.999, lambda_L1=100.)
    model = init_model(model, DEVICE)

    # Loss meters
    loss_meters = create_loss_meters()

    # Initialize FID
    metrics_calculator = Metrics(DEVICE)

    best_loss = float('inf')

    fids = []
    ssims = []
    psnrs = []

    # Training loop
    print(f"[INFO] Training starting ...")
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        # Train for one epoch
        train_epoch(model, train_loader, loss_meters)

        # Log results
        print(f"[INFO] Epoch [{epoch + 1}/{NUM_EPOCHS}] - Time: {time.time() - start_time:.2f}s")
        log_results(loss_meters)

        # Visualize results
        if (epoch) % VISUALIZE_INTERVAL == 0:
            visualize(model, next(iter(val_loader)), SAVE_DIR, epoch, save=True)

        if (epoch) % METRICS_INTERVAL == 0:
            print(f"[INFO] Validation starting ...")

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

            print(f"\n[INFO] Epoch [{epoch + 1}/{NUM_EPOCHS}] Results:")
            print(f"{'-'*50}")
            print(f"  FID Score   : {np.mean(fids):.4f}")
            print(f"  Mean SSIM   : {np.mean(ssims):.4f}")
            print(f"  Mean PSNR   : {np.mean(psnrs):.4f} dB")
            print(f"{'-'*50}\n")


            print(f"[INFO] Validation Ends.")
        
        # Save best model and checkpoint models
        best_loss = save_model(model, best_loss, loss_meters['generator_loss'].avg, SAVE_DIR, MODEL_NAME)

        # Reset loss meters
        for meter in loss_meters.values():
            meter.reset()

    print("Training complete.")

if __name__ == "__main__":
    main()
