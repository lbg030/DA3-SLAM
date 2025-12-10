import torch
import numpy as np
import matplotlib.pyplot as plt

def show_plot(frame, title=None, cmap="jet",figsize=[5,5]):    
    plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title)
    if isinstance(frame, torch.Tensor):
        frame = frame.squeeze().float()
        plt.imshow(frame.detach().cpu().numpy(), cmap=cmap) if frame.shape[0] != 3 else plt.imshow(frame.permute(1,2,0).detach().cpu().numpy())
    
    elif isinstance(frame, np.ndarray):
        frame = frame.astype(np.float32)
        plt.imshow(frame, cmap=cmap) if frame.shape[0] != 3 else plt.imshow(frame.transpose(1,2,0)) 
        
    else :
        assert False, "Invalid frame type"
        