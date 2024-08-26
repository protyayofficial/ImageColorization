import torch
import torch.nn as nn

def init_weights(net, init_type='norm', gain=0.02):
    """
    Initialize network weights.

    Parameters:
    - net (nn.Module): The neural network whose weights need to be initialized.
    - init_type (str): Type of initialization - 'norm' for normal distribution, 'xavier' for Xavier initialization, 
                       or 'kaiming' for Kaiming initialization. Default is 'norm'.
    - gain (float): Scaling factor for initialization. Default is 0.02.
    """
    
    def init_func(m):
        """
        Apply the specified initialization function to the model layers.

        Parameters:
        - m (nn.Module): Model layer to be initialized.
        """
        classname = m.__class__.__name__
        
        # Initialize Conv layers
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init_type == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0.0, mode='fan_in')
            else:
                raise ValueError(f"Initialization method '{init_type}' not recognized.")
            
            # Initialize biases if present
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        
        # Initialize BatchNorm layers
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, mean=1.0, std=gain)
            nn.init.constant_(m.bias.data, 0.0)
    
    # Apply initialization
    net.apply(init_func)
    print(f"[INFO] Model {type(net).__name__} Initialized with {init_type} initialization")

    return net

def init_model(model, device, init_type='norm', gain=0.02):
    """
    Initialize the model with specified weight initialization and move to the device.

    Parameters:
    - model (nn.Module): The model to be initialized and moved to the device.
    - device (torch.device): The device where the model will be moved (e.g., 'cuda' or 'cpu').
    - init_type (str): Type of initialization for the weights. Default is 'norm'.
    - gain (float): Scaling factor for the initialization. Default is 0.02.

    Returns:
    - nn.Module: The initialized model on the specified device.
    """
    # Move model to the specified device
    model = model.to(device)
    
    # Initialize model weights
    model = init_weights(model, init_type=init_type, gain=gain)
    
    return model
