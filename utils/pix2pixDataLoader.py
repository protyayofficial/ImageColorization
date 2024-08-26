from PIL import Image
import numpy as np
from skimage.color import rgb2lab

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Constants
IMG_SIZE = 256

class ColorizationDataset(Dataset):
    """
    Custom dataset class for image colorization.

    Parameters:
    - paths (list): List of file paths to the images.
    - split (str): 'train' for training data or 'val' for validation data. 
                   This determines the applied transformations.
    """
    def __init__(self, paths, split='train'):
        self.paths = paths
        self.size = IMG_SIZE
        self.split = split

        # Define transformations for training and validation
        if split == "train":
            self.transforms = transforms.Compose([
                transforms.Resize((self.size, self.size)),
                transforms.RandomHorizontalFlip()
            ])
        elif split == "val":
            self.transforms = transforms.Resize((self.size, self.size), Image.BICUBIC)
        else:
            raise ValueError(f"Split '{split}' not recognized. Use 'train' or 'val'.")

    def __getitem__(self, idx):
        """
        Retrieves an item by index and applies necessary transformations and LAB color space conversion.

        Parameters:
        - idx (int): Index of the image in the dataset.

        Returns:
        - dict: A dictionary with 'L' (grayscale) and 'ab' (color channels) tensors.
        """
        # Load image and convert to RGB
        img = Image.open(self.paths[idx]).convert('RGB')

        # Apply transformations
        img = self.transforms(img)
        img = np.array(img)

        # Convert RGB to LAB color space
        img_lab = rgb2lab(img).astype('float32')

        # Convert to tensor and normalize
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1.   # L channel normalized to [-1, 1]
        ab = img_lab[[1, 2], ...] / 110.    # ab channels normalized to [-1, 1]

        return {'L': L, 'ab': ab}

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.paths)


def make_dataloaders(paths, batch_size=32, num_workers=4, pin_memory=True, split='train'):
    """
    Utility function to create dataloaders for training or validation.

    Parameters:
    - paths (list): List of file paths to the images.
    - batch_size (int): Number of samples per batch. Default is 32.
    - num_workers (int): Number of subprocesses to use for data loading. Default is 4.
    - pin_memory (bool): Whether to use pinned (page-locked) memory. Default is True.
    - split (str): Either 'train' or 'val' for training or validation data.

    Returns:
    - DataLoader: PyTorch DataLoader for the given dataset.
    """
    # Create dataset instance
    dataset = ColorizationDataset(paths=paths, split=split)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        shuffle=(split == 'train')  # Only shuffle for training
    )

    return dataloader


