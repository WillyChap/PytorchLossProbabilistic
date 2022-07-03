"""
Author: WE Chapman; wchapman@ucar.edu
date: Jul 1. 2022

See Links below for CDF and PDF (used in calc.) and general knoweldge tips on uncertainty estimation

details: Laplace distribution - https://en.wikipedia.org/wiki/Laplace_distribution
details: Normal distribution - https://en.wikipedia.org/wiki/Normal_distribution
details: Cauchy distribution - https://en.wikipedia.org/wiki/Cauchy_distribution
details: CRPS - Gauss - https://journals.ametsoc.org/view/journals/wefo/15/5/1520-0434_2000_015_0559_dotcrp_2_0_co_2.xml
details: GaunssNLL - https://stats.stackexchange.com/questions/521091/optimizing-gaussian-negative-log-likelihood

Important: your NN should have two output streams (or more depending on the distribution) 
see: https://github.com/WillyChap/NonLIMear/deeplim/readout_MLP.py -- for an example (function : LIM_MLP_GaussLL() )

NLL/CRPS as a proper scoring rules (FIND MORE DISTRIBUTIONS TO CODE UP PARAMETRICALLY HERE): 
- https://www.jstatsoft.org/article/view/v090i12

Dangers of using thresholding on Evaluating probabilistic methods (and work arounds):
- https://projecteuclid.org/journals/statistical-science/volume-32/issue-1/Forecasters-Dilemma-Extreme-Events-and-Forecast-Evaluation/10.1214/16-STS588.full
- https://journals.ametsoc.org/view/journals/mwre/145/9/mwr-d-16-0487.1.xml
- https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.3380
- https://www.tandfonline.com/doi/full/10.3402/tellusa.v65i0.21206

Excellent Figure explaining CRPS-- See Figure 2 of : https://journals.ametsoc.org/jcli/article/32/1/161/89277/Precipitation-Prediction-Skill-for-the-West-Coast

pytorch code/class inspired by: https://deebuls.github.io/devblog/
"""

def get_loss(name, reduction='mean'):
    # Specify loss function
    name = name.lower().strip()
    if name in ['l1', 'mae']:
        loss = nn.L1Loss(reduction=reduction)
    elif name in ['l2', 'mse']:
        loss = nn.MSELoss(reduction=reduction)  
    elif name in ['gauss']:
        print('getttt probable babbby')
        loss = nn.GaussianNLLLoss()
    elif name in ['laplace']:
        print('getttt probable babbby')
        loss = Custom_Laplace()
    elif name in ['cauchy']:
        print('getttt probable babbby')
        loss = Custom_Cauchy()
    elif name in ['crps']:
        print('getttt probable babbby')
        loss = Custom_CRPS()
    else:
        raise ValueError('Available Losses: MAE, L1, L2, MSE, Gauss, Laplace, Cauchy, CRPS ... ')  # default
    return loss


class Custom_CRPS(nn.Module):
    """
    compute the CRPS cost function of a normal distribution defined by the
    mean and std. 
    
    Args: 
        input: mean value
        target: observed value
        scale: standard deviation (estimated)
    
    Returns: 
        Gaussian CRPS: Scalar with CRPS over the batch
    
    """
    def __init__(self):
        super(Custom_CRPS,self).__init__();
        
    def forward(self,input, target, scale, eps=1e-06, reduction='mean'):
        # Inputs and targets much have same shape
        input = input.view(input.size(0), -1)
        target = target.view(target.size(0), -1)
        if input.size() != target.size():
              raise ValueError("input and target must have same size")

        # Second dim of scale must match that of input or be equal to 1
        scale = scale.view(input.size(0), -1)
        if scale.size(1) != input.size(1) and scale.size(1) != 1:
            raise ValueError("scale is of incorrect size")

        # Check validity of reduction mode
        if reduction != 'none' and reduction != 'mean' and reduction != 'sum':
            raise ValueError(reduction + " is not valid")

        # Entries of var must be non-negative
        if torch.any(scale < 0):
            raise ValueError("scale has negative entry/entries")

        # Clamp for stability
        scale = scale.clone()
        with torch.no_grad():
            scale.clamp_(min=eps)

        # Calculate loss (without constant)
        loc =(target-input)/scale
        pie = torch.as_tensor(math.pi) #yummmm
        phi =1.0 / torch.sqrt((2.0*pie))*torch.exp(-torch.square(loc)/2.0)
        Phi = 0.5*(1.0+torch.erf((loc/torch.sqrt(torch.as_tensor(2.0)))))
        loss = scale * (loc * (2.0 * Phi - 1.) + 2.0 * phi - 1.0 / torch.sqrt(pie))

        # Apply reduction
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
def CRPSloss(input, target, scale, eps=1e-06, reduction='mean'):
    """
    compute the CRPS cost function of a normal distribution defined by the
    mean and std. 
    
    Args: 
        input: mean value
        target: observed value
        scale: standard deviation (estimated)
    
    Returns: 
        Gaussian CRPS: Scalar with CRPS over the batch
    
    """
    # Inputs and targets much have same shape
    input = input.view(input.size(0), -1)
    target = target.view(target.size(0), -1)
    if input.size() != target.size():
          raise ValueError("input and target must have same size")

    # Second dim of scale must match that of input or be equal to 1
    scale = scale.view(input.size(0), -1)
    if scale.size(1) != input.size(1) and scale.size(1) != 1:
        raise ValueError("scale is of incorrect size")

    # Check validity of reduction mode
    if reduction != 'none' and reduction != 'mean' and reduction != 'sum':
        raise ValueError(reduction + " is not valid")

    # Entries of var must be non-negative
    if torch.any(scale < 0):
        raise ValueError("scale has negative entry/entries")

    # Clamp for stability
    scale = scale.clone()
    with torch.no_grad():
        scale.clamp_(min=eps)

    # Calculate loss (without constant)
    loc =(target-input)/scale
    pie = torch.as_tensor(math.pi) #yummmm
    phi =1.0 / torch.sqrt((2.0*pie))*torch.exp(-torch.square(loc)/2.0)
    Phi = 0.5*(1.0+torch.erf((loc/torch.sqrt(torch.as_tensor(2.0)))))
    loss = scale * (loc * (2.0 * Phi - 1.) + 2.0 * phi - 1.0 / torch.sqrt(pie))

    # Apply reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
    

class Custom_Laplace(nn.Module):
    """
    compute the Negative Log Liklihood cost function of a laplace distribution defined by the
    mean and std. 
    
    Args: 
        input: mean value
        target: observed value
        scale: standard deviation (estimated)
    
    Returns: 
        laplace NLL loss
    
    """
    def __init__(self):
        super(Custom_Laplace,self).__init__();
        
    def forward(self,input, target, scale, eps=1e-06, reduction='mean'):
        loss = torch.log(2*scale) + torch.abs(input - target)/scale

        # Inputs and targets much have same shape
        input = input.view(input.size(0), -1)
        target = target.view(target.size(0), -1)
        if input.size() != target.size():
              raise ValueError("input and target must have same size")

        # Second dim of scale must match that of input or be equal to 1
        scale = scale.view(input.size(0), -1)
        if scale.size(1) != input.size(1) and scale.size(1) != 1:
            raise ValueError("scale is of incorrect size")

        # Check validity of reduction mode
        if reduction != 'none' and reduction != 'mean' and reduction != 'sum':
            raise ValueError(reduction + " is not valid")

        # Entries of var must be non-negative
        if torch.any(scale < 0):
            raise ValueError("scale has negative entry/entries")

        # Clamp for stability
        scale = scale.clone()
        with torch.no_grad():
            scale.clamp_(min=eps)

        # Calculate loss (without constant)
        loss = (torch.log(2*scale) + torch.abs(input - target) / scale).view(input.size(0), -1).sum(dim=1)

        # Apply reduction
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
    
def LaplaceNLLLoss(input, target, scale, eps=1e-06, reduction='mean'):
    """
    compute the Negative Log Liklihood cost function of a laplace distribution defined by the
    mean and std. 
    
    Args: 
        input: mean value
        target: observed value
        scale: standard deviation (estimated)
    
    Returns: 
        laplace NLL loss
    
    """
    loss = torch.log(2*scale) + torch.abs(input - target)/scale

    # Inputs and targets much have same shape
    input = input.view(input.size(0), -1)
    target = target.view(target.size(0), -1)
    if input.size() != target.size():
          raise ValueError("input and target must have same size")

    # Second dim of scale must match that of input or be equal to 1
    scale = scale.view(input.size(0), -1)
    if scale.size(1) != input.size(1) and scale.size(1) != 1:
        raise ValueError("scale is of incorrect size")

    # Check validity of reduction mode
    if reduction != 'none' and reduction != 'mean' and reduction != 'sum':
        raise ValueError(reduction + " is not valid")

    # Entries of var must be non-negative
    if torch.any(scale < 0):
        raise ValueError("scale has negative entry/entries")

    # Clamp for stability
    scale = scale.clone()
    with torch.no_grad():
        scale.clamp_(min=eps)

    # Calculate loss (without constant)
    loss = (torch.log(2*scale) + torch.abs(input - target) / scale).view(input.size(0), -1).sum(dim=1)

    # Apply reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
    
    
class Custom_Cauchy(nn.Module):
    """
    compute the Negative Log Liklihood cost function of a Cauchy distribution defined by the
    mean and std. 
    
    Args: 
        input: mean value
        target: observed value
        scale: standard deviation (estimated)
    
    Returns: 
        Cauchy NLL loss
    """
    
    def __init__(self):
        super(Custom_Cauchy,self).__init__();
        
    def forward(self,input, target, scale, eps=1e-06, reduction='mean'):
        # Inputs and targets much have same shape
        input = input.view(input.size(0), -1)
        target = target.view(target.size(0), -1)
        if input.size() != target.size():
            raise ValueError("input and target must have same size")

        # Second dim of scale must match that of input or be equal to 1
        scale = scale.view(input.size(0), -1)
        if scale.size(1) != input.size(1) and scale.size(1) != 1:
            raise ValueError("scale is of incorrect size")

        # Check validity of reduction mode
        if reduction != 'none' and reduction != 'mean' and reduction != 'sum':
            raise ValueError(reduction + " is not valid")

        # Entries of var must be non-negative
        if torch.any(scale < 0):
            raise ValueError("scale has negative entry/entries")

        # Clamp for stability
        scale = scale.clone()
        with torch.no_grad():
            scale.clamp_(min=eps)

        # Calculate loss (without constant)
        loss = (torch.log(3.14159265*scale) + torch.log(1 + ((input - target)**2)/scale**2)) .view(input.size(0), -1).sum(dim=1)

        # Apply reduction
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss
   
    
def CauchyNLLLoss(input, target, scale, eps=1e-06, reduction='mean'):
    """
    compute the Negative Log Liklihood cost function of a Cauchy distribution defined by the
    mean and std. 
    
    Args: 
        input: median value
        target: observed value
        scale: standard deviation (estimated)
    
    Returns: 
        Cauchy NLL loss
    """
    # Inputs and targets much have same shape
    input = input.view(input.size(0), -1)
    target = target.view(target.size(0), -1)
    if input.size() != target.size():
        raise ValueError("input and target must have same size")

    # Second dim of scale must match that of input or be equal to 1
    scale = scale.view(input.size(0), -1)
    if scale.size(1) != input.size(1) and scale.size(1) != 1:
        raise ValueError("scale is of incorrect size")

    # Check validity of reduction mode
    if reduction != 'none' and reduction != 'mean' and reduction != 'sum':
        raise ValueError(reduction + " is not valid")

    # Entries of var must be non-negative
    if torch.any(scale < 0):
        raise ValueError("scale has negative entry/entries")

    # Clamp for stability
    scale = scale.clone()
    with torch.no_grad():
        scale.clamp_(min=eps)

    # Calculate loss (without constant)
    loss = (torch.log(3.14159265*scale) + torch.log(1 + ((input - target)**2)/scale**2)) .view(input.size(0), -1).sum(dim=1)


    # Apply reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
