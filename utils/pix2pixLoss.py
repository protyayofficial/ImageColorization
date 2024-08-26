import torch
import torch.nn as nn

class GANLoss(nn.Module):
    """
    Define different GAN loss functions.

    Parameters:
    - gan_mode (str): Type of GAN loss. Options are 'vanilla' for BCE loss and 'lsgan' for MSE loss.
    - real_label (float): Label value for real images. Default is 1.0.
    - fake_label (float): Label value for fake images. Default is 0.0.
    """

    def __init__(self, gan_mode="vanilla", real_label=1.0, fake_label=0.0):
        super(GANLoss, self).__init__()
        
        # Register label values for real and fake images as buffers
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))

        # Choose loss function based on gan_mode
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        else:
            raise ValueError(f"GAN mode '{gan_mode}' not recognized. Use 'vanilla' or 'lsgan'.")

    def get_labels(self, preds, target_is_real):
        """
        Generate labels for the input.

        Parameters:
        - preds (Tensor): Predictions from the discriminator.
        - target_is_real (bool): Whether the target is real or fake.

        Returns:
        - Tensor: Label tensor with the same size as preds.
        """
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        
        return labels.expand_as(preds)
    
    def forward(self, preds, target_is_real):
        """
        Forward pass to calculate loss.

        Parameters:
        - preds (Tensor): Predictions from the discriminator.
        - target_is_real (bool): Whether the target is real or fake.

        Returns:
        - Tensor: Calculated GAN loss.
        """
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss
