import torch

def default_loss(x, eps_pred, eps, criterion):
    '''Calculate the default loss for the ddpm'''
        
    loss = criterion(eps_pred, eps)

    return loss, loss

def iso_loss(x, eps_pred, eps, criterion):
    '''Calculate the isotropy loss for the ddpm'''
        
    loss = criterion(eps_pred, eps)
    
    squared_trace_eps = torch.mean(torch.sum(eps_pred**2, dim=2))

    normalized_squared_trace_eps = squared_trace_eps / torch.tensor(2.0, requires_grad=False)

    norm_loss = 1.0 - normalized_squared_trace_eps

    return loss, norm_loss