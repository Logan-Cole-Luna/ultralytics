import math
import torch
from torch.optim.optimizer import Optimizer

# New helper for momentum update
def momentum_update(state, grad, momentum):
    if 'momentum_buffer' not in state:
        state['momentum_buffer'] = grad.clone().detach()
    else:
        state['momentum_buffer'].mul_(momentum).add_(grad)
    return state['momentum_buffer']


class GroupGradientDescent(Optimizer):
    """
    Implements Group Gradient Descent optimization with optional momentum, adaptive scaling, and weight decay.
    
    This optimizer normalizes gradients within groups of parameters, helping to balance
    the scale of updates across different layers of the network.

    Args:
        params (iterable): Iterable of parameters to optimize
        lr (float, optional): Learning rate (default: 1e-3)
        group_size (int, optional): Fixed number of parameters per group. If None, computed dynamically.
        num_groups (int, optional): If provided, divides gradients into exactly this many groups.
                                    If None, computed dynamically.
        eps (float, optional): Small constant for numerical stability (default: 1e-5)
        momentum (float, optional): Momentum factor (default: 0)
        adaptive (bool, optional): Whether to use adaptive learning rate scaling (default: False)
        adaptive_eps (float, optional): Small constant for adaptive scaling stability (default: 1e-8)
        weight_decay (float, optional): Weight decay factor for L2 regularization (default: 0)

    Algorithm:
        For each group of parameters:
        1. Reshape gradients into groups of size 'group_size'
        2. Normalize each group using its mean and standard deviation
        3. Apply momentum if specified
        4. Apply adaptive scaling if enabled
        5. Apply weight decay for regularization
        6. Update parameters using normalized gradients
    """
    def __init__(self, params, lr=1e-3, group_size=None, num_groups=None, eps=1e-5, momentum=0, adaptive=False, adaptive_eps=1e-8, weight_decay=0):
        # Initialize optimizer with learning rate (lr), number of parameters per group (group_size),
        # small constant (eps) to prevent division by zero, momentum, and weight decay
        defaults = dict(lr=lr, group_size=group_size, num_groups=num_groups, eps=eps, momentum=momentum, adaptive=adaptive, adaptive_eps=adaptive_eps, weight_decay=weight_decay)
        super(GroupGradientDescent, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            eps = group['eps']
            momentum = group.get('momentum', 0)
            adaptive = group.get('adaptive', False)
            adaptive_eps = group.get('adaptive_eps', 1e-8)
            weight_decay = group.get('weight_decay', 0)
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                # Convert n-dimensional gradient tensor into 1D vector
                flat_grad = grad.view(-1)
                N = flat_grad.numel()
                
                # Determine group_size using provided values or dynamic selection
                given_group_size = group.get('group_size')
                given_num_groups  = group.get('num_groups')
                if given_group_size is not None:
                    group_size = given_group_size
                elif given_num_groups is not None:
                    group_size = math.ceil(N / given_num_groups)
                else:
                    dynamic_num_groups = max(1, int(math.sqrt(N)))
                    group_size = math.ceil(N / dynamic_num_groups)
                
                remainder = N % group_size

                # If N is not perfectly divisible by group_size, pad with zeros
                # This ensures we can reshape into complete groups
                if remainder != 0:
                    pad_size = group_size - remainder
                    flat_grad = torch.cat([flat_grad, flat_grad.new_zeros(pad_size)], dim=0)

                # Reshape into matrix of shape (⌈N/group_size⌉, group_size)
                # Each row represents one group of parameters
                reshaped = flat_grad.view(-1, group_size)

                # Calculate mean μᵢ for each group i along group dimension
                # Shape: (num_groups, 1)
                group_mean = reshaped.mean(dim=1, keepdim=True)

                # Calculate standard deviation σᵢ for each group i along group dimension
                # Add eps to prevent division by zero
                # σᵢ = sqrt(1/n ∑(x - μᵢ)²) + ε
                # Shape: (num_groups, 1)
                group_std = reshaped.std(dim=1, keepdim=True) + eps

                # Normalize each group using its mean and std
                # For each element x in group i: x' = (x - μᵢ)/(σᵢ)
                normalized = (reshaped - group_mean) / group_std

                # Convert back to 1D and remove any padding we added
                # Only keep the first N elements (original gradient size)
                normalized_flat_grad = normalized.view(-1)[:N]

                # Reshape normalized gradient back to original parameter shape
                normalized_grad = normalized_flat_grad.view_as(p.data)
                
                # Apply weight decay for regularization
                if weight_decay:
                    #pass
                    normalized_grad = normalized_grad + weight_decay * p.data
                
                # Get optimizer state
                state = self.state[p]
                
                # Apply momentum update (using helper from misc.utils)
                buf = momentum_update(state, normalized_grad, momentum)
                
                # Handle adaptive scaling if enabled
                if adaptive:
                    if 'sum_sq_grad' not in state:
                        state['sum_sq_grad'] = buf.detach().pow(2)
                    else:
                        state['sum_sq_grad'].add_(buf.detach().pow(2))
                    adaptive_buf = buf / (state['sum_sq_grad'].sqrt() + adaptive_eps)
                else:
                    adaptive_buf = buf

                # Update parameter using the momentum buffer
                p.data.add_(adaptive_buf, alpha=-lr)
        return loss
