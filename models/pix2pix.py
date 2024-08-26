import torch
import torch.nn as nn
import torch.optim as optim

from utils.pix2pixInitWeights import init_model
from utils.pix2pixLoss import GANLoss

class UNetBlock(nn.Module):
    """
    Defines a single block in the UNet architecture with downsampling (conv) and upsampling (transposed conv).
    
    Args:
        out_filters (int): Number of output filters for the upsampling layer.
        in_filters (int): Number of input filters for the downsampling layer.
        submodule (nn.Module, optional): The submodule that will be nested in this block. Default is None.
        input_channels (int, optional): Number of input channels. Default is None.
        dropout (bool, optional): Whether to include dropout in the block. Default is False.
        innermost (bool, optional): Whether this is the innermost block. Default is False.
        outermost (bool, optional): Whether this is the outermost block. Default is False.
    """
    def __init__(self, out_filters, in_filters, submodule=None, input_channels=None, dropout=False, innermost=False, outermost=False):
        super().__init__()

        self.outermost = outermost

        if input_channels is None:
            input_channels = out_filters

        # Downsampling layers
        downconv = nn.Conv2d(in_channels=input_channels, out_channels=in_filters, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        downnorm = nn.BatchNorm2d(num_features=in_filters)

        # Upsampling layers
        uprelu = nn.ReLU(inplace=True)
        upnorm = nn.BatchNorm2d(num_features=out_filters)

        if outermost:
            # Outermost block doesn't have normalization and uses Tanh activation for the final output
            upconv = nn.ConvTranspose2d(in_channels=in_filters * 2, out_channels=out_filters, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up

        elif innermost:
            # Innermost block only downsamples and upsamples without concatenation
            upconv = nn.ConvTranspose2d(in_channels=in_filters, out_channels=out_filters, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up

        else:
            # Intermediate blocks have both downsampling and upsampling, including concatenation and optional dropout
            upconv = nn.ConvTranspose2d(in_channels=in_filters * 2, out_channels=out_filters, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if dropout:
                up += [nn.Dropout(0.5)]
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        # Forward propagation through the block
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], dim=1)
        

class UNet(nn.Module):
    """
    UNet architecture for the generator, typically used in image translation tasks.
    
    Args:
        input_channels (int): Number of input channels (e.g., grayscale images = 1 channel).
        output_channels (int): Number of output channels (e.g., 2 channels for ab color channels).
        num_downsampling (int): Number of downsampling layers.
        num_init_filters (int): Number of initial filters (doubles with each downsampling).
    """
    def __init__(self, input_channels=1, output_channels=2, num_downsampling=8, num_init_filters=64):
        super().__init__()

        # Construct the UNet by nesting UNet blocks
        unet_block = UNetBlock(out_filters=num_init_filters * 8, in_filters=num_init_filters * 8, innermost=True)

        # Add intermediate downsampling blocks with dropout
        for _ in range(num_downsampling - 5):
            unet_block = UNetBlock(out_filters=num_init_filters * 8, in_filters=num_init_filters * 8, submodule=unet_block, dropout=True)

        # Add intermediate blocks that gradually decrease filter sizes
        out_filters = num_init_filters * 8
        for _ in range(3):
            unet_block = UNetBlock(out_filters=out_filters // 2, in_filters=out_filters, submodule=unet_block)
            out_filters //= 2

        # Final block that outputs the desired number of channels
        self.model = UNetBlock(out_filters=output_channels, in_filters=out_filters, submodule=unet_block, input_channels=input_channels, outermost=True)

    def forward(self, x):
        # Forward pass through the UNet model
        return self.model(x)
    

class PatchDiscriminator(nn.Module):
    """
    PatchGAN Discriminator for assessing patches of the input image. This model classifies overlapping image patches as real or fake.
    
    Args:
        input_channels (int): Number of input channels (e.g., grayscale + ab color channels).
        num_init_filters (int): Initial number of filters.
        num_downsampling (int): Number of downsampling layers in the network.
    """
    def __init__(self, input_channels, num_init_filters=64, num_downsampling=3):
        super().__init__()

        # Define the layers of the PatchDiscriminator using helper method
        model = [self.get_layers(input_filters=input_channels, output_filters=num_init_filters, normalization=False)]

        # Add downsampling layers
        model += [self.get_layers(input_filters=num_init_filters * 2 ** i, output_filters=num_init_filters * 2 ** (i + 1), stride=1 if i == (num_downsampling - 1) else 2) for i in range(num_downsampling)]

        # Final layer outputs 1 channel per patch (binary classification of real vs. fake)
        model += [self.get_layers(input_filters=num_init_filters * 2 ** num_downsampling, output_filters=1, stride=1, normalization=False, activation=False)]

        self.model = nn.Sequential(*model)


    def get_layers(self, input_filters, output_filters, kernel_size=4, stride=2, padding=1, normalization=True, activation=True):
        """
        Helper function to create convolutional layers for PatchDiscriminator.
        
        Args:
            input_filters (int): Number of input filters.
            output_filters (int): Number of output filters.
            kernel_size (int): Kernel size for convolution.
            stride (int): Stride for convolution.
            padding (int): Padding for convolution.
            normalization (bool): Whether to include BatchNorm after convolution.
            activation (bool): Whether to include activation after convolution.
        """
        layers = [nn.Conv2d(in_channels=input_filters, out_channels=output_filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=not normalization)]
        
        if normalization:
            layers += [nn.BatchNorm2d(num_features=output_filters)]

        if activation:
            layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]

        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Forward pass through the PatchDiscriminator
        return self.model(x)


        
class MainModel(nn.Module):
    """
    Main model class that integrates the generator (UNet) and discriminator (PatchGAN), handles the training loops, and calculates the loss functions.
    
    Args:
        generator_model (nn.Module): The generator model (optional, defaults to UNet).
        generator_lr (float): Learning rate for the generator.
        discriminator_lr (float): Learning rate for the discriminator.
        beta1 (float): Beta1 hyperparameter for Adam optimizer.
        beta2 (float): Beta2 hyperparameter for Adam optimizer.
        lambda_L1 (float): Weight for the L1 loss in the generator's objective.
    """
    def __init__(self, generator_model=None, generator_lr=2e-4, discriminator_lr=2e-4, beta1=0.5, beta2=0.999, lambda_L1=100.):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1

        # Initialize generator model (UNet by default)
        if generator_model is None:
            self.generator_model = init_model(model=UNet(input_channels=1, output_channels=2, num_downsampling=8, num_init_filters=64), device=self.device)
        else:
            self.generator_model = generator_model.to(self.device)

        # Initialize PatchGAN Discriminator
        self.discriminator_model = init_model(model=PatchDiscriminator(input_channels=3, num_downsampling=3, num_init_filters=64), device=self.device)

        # Define loss functions
        self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1criterion = nn.L1Loss()

        # Define optimizers
        self.generator_optimizer = optim.Adam(self.generator_model.parameters(), lr=generator_lr, betas=(beta1, beta2))
        self.discriminator_optimizer = optim.Adam(self.discriminator_model.parameters(), lr=discriminator_lr, betas=(beta1, beta2))

    def set_required_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def setup_input(self, data):
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)

    def forward(self):
        """
        Forward pass through the generator.
        
        Args:
            input_image (torch.Tensor): The grayscale input image.
            
        Returns:
            torch.Tensor: The colorized output image (ab channels).
        """
        self.fake_color = self.generator_model(self.L)
    
    def backward_discriminator(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.discriminator_model(fake_image.detach())
        self.discriminator_loss_fake = self.GANcriterion(fake_preds, False)

        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.discriminator_model(real_image)
        self.discriminator_loss_real = self.GANcriterion(real_preds, True)

        self.discriminator_loss = (self.discriminator_loss_fake  + self.discriminator_loss_real) * 0.5

        self.discriminator_loss.backward()

    def backward_generator(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.discriminator_model(fake_image)
        
        self.generator_loss_GAN = self.GANcriterion(fake_preds, True)
        self.generator_loss_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        
        self.generator_loss = self.generator_loss_GAN + self.generator_loss_L1

        self.generator_loss.backward()

    def optimize(self):
        self.forward()

        self.discriminator_model.train()
        self.set_required_grad(self.discriminator_model, True)
        self.discriminator_optimizer.zero_grad()
        self.backward_discriminator()
        self.discriminator_optimizer.step()

        self.generator_model.train()
        self.set_required_grad(self.discriminator_model, False)
        self.generator_optimizer.zero_grad()
        self.backward_generator()
        self.generator_optimizer.step()

    def save_models(self, generator_path, discriminator_path):
        """
        Save the generator and discriminator models.
        
        Args:
            generator_path (str): Path to save the generator model.
            discriminator_path (str): Path to save the discriminator model.
        """
        torch.save(self.generator_model.state_dict(), generator_path)
        torch.save(self.discriminator_model.state_dict(), discriminator_path)

