import math
import torch
from torch.optim.optimizer import Optimizer

# New helper for momentum update
def momentum_update(state, grad, momentum):
    """
    Apply momentum update to gradients.
    
    For parameter with gradient g at iteration t:
    v_t = momentum * v_{t-1} + g_t
    where v_t is the momentum buffer
    
    Args:
        state: Optimizer state dictionary for the parameter
        grad: Current gradient
        momentum: Momentum coefficient (β)
        
    Returns:
        Updated momentum buffer
    """
    if 'momentum_buffer' not in state:
        state['momentum_buffer'] = grad.clone().detach()
    else:
        state['momentum_buffer'].mul_(momentum).add_(grad)  # v_t = β*v_{t-1} + g_t
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
        layer_wise (bool, optional): Whether to use layer-wise grouping instead of fixed group size (default: False)
                                    When enabled, parameters are grouped by their layer instead of arbitrary fixed sizes.
        clip_grad_norm (float, optional): Value to clip gradient norm to, disabled if None (default: None)
                                         Helps prevent exploding gradients.
        adaptive_lr (bool, optional): Whether to use adaptive learning rate (default: False)
                                     Adjusts learning rate based on the square root of sum of squared gradients.

    Mathematical formulation:
        1. Group parameters (by fixed size or by layer)
        2. For each group i:
           - Compute mean μᵢ and standard deviation σᵢ
           - Normalize: g' = (g - μᵢ)/(σᵢ + ε)
        3. Apply momentum: v_t = β*v_{t-1} + g'
        4. Apply adaptive scaling (if enabled): v'_t = v_t / sqrt(Σ v_t^2 + ε)
        5. Apply weight decay: g' = g' + λ*w
        6. Update parameters: w_t = w_{t-1} - η*v'_t
           (where η may be adjusted by adaptive_lr if enabled)
    """
    def __init__(self, params, lr=1e-3, group_size=None, num_groups=None, eps=1e-5, 
                 momentum=0, adaptive=False, adaptive_eps=1e-8, weight_decay=0,
                 layer_wise=False, clip_grad_norm=None, adaptive_lr=False):
        defaults = dict(lr=lr, group_size=group_size, num_groups=num_groups, eps=eps, 
                        momentum=momentum, adaptive=adaptive, adaptive_eps=adaptive_eps, 
                        weight_decay=weight_decay, layer_wise=layer_wise,
                        clip_grad_norm=clip_grad_norm, adaptive_lr=adaptive_lr)
        super(GroupGradientDescent, self).__init__(params, defaults)
        
        # If layer_wise is True, organize parameters by layer
        if layer_wise:
            self._organize_layer_groups()
    
    def _organize_layer_groups(self):
        """
        Organize parameters into layer groups based on parameter names.
        
        This method creates a mapping from parameter to its layer index by:
        1. Collecting all unique layer names from parameter names
        2. Creating a mapping from layer name to index
        3. Assigning each parameter to its corresponding layer index
        
        This enables layer-wise normalization where gradients from the same 
        logical layer are normalized together, preserving intra-layer relationships.
        """
        self.param_to_layer = {}
        layer_names = set()
        
        # First, collect all unique layer names
        for group in self.param_groups:
            for p in group['params']:
                if hasattr(p, '_name') and p._name is not None:
                    # Extract layer name from parameter name (e.g., 'conv1.weight' -> 'conv1')
                    layer_name = p._name.split('.')[0]
                    layer_names.add(layer_name)
        
        # Create a mapping from layer name to index
        # Sorting ensures consistent indexing across runs
        layer_indices = {name: idx for idx, name in enumerate(sorted(layer_names))}
        
        # Map each parameter to its layer index
        for group in self.param_groups:
            for p in group['params']:
                if hasattr(p, '_name') and p._name is not None:
                    layer_name = p._name.split('.')[0]
                    self.param_to_layer[p] = layer_indices.get(layer_name, 0)
                else:
                    # Default layer index for parameters without names
                    self.param_to_layer[p] = 0

    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
                
        Returns:
            loss: The loss value returned by the closure, or None if closure is None.
        """
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
            layer_wise = group.get('layer_wise', False)
            clip_grad_norm = group.get('clip_grad_norm')
            adaptive_lr_flag = group.get('adaptive_lr', False)
            
            if layer_wise:
                # For layer-wise grouping, process each layer separately
                layer_params = {}
                for p in group['params']:
                    if p.grad is None:
                        continue
                    
                    # Apply gradient clipping if specified
                    # Helps prevent exploding gradients by limiting the L2 norm
                    if clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(p, clip_grad_norm)
                    
                    # Get layer index for this parameter and group parameters by layer
                    layer_idx = self.param_to_layer.get(p, 0)
                    if layer_idx not in layer_params:
                        layer_params[layer_idx] = []
                    layer_params[layer_idx].append(p)
                
                # Process each layer
                for layer_idx, params in layer_params.items():
                    # Collect all gradients in this layer
                    all_grads = []
                    for p in params:
                        all_grads.append(p.grad.data.view(-1))
                    
                    # Concatenate all gradients from this layer into a single tensor
                    if all_grads:
                        layer_grad = torch.cat(all_grads)
                        
                        # Normalize the entire layer's gradients
                        # μ = mean of all gradients in the layer
                        layer_mean = layer_grad.mean()
                        # σ = standard deviation of all gradients in the layer
                        layer_std = layer_grad.std() + eps  # Add eps for numerical stability
                        # g' = (g - μ)/σ for all gradients in the layer
                        normalized_layer_grad = (layer_grad - layer_mean) / layer_std
                        
                        # Distribute normalized gradients back to parameters
                        start_idx = 0
                        for p in params:
                            grad_size = p.grad.data.numel()
                            # Extract the portion of normalized gradients corresponding to this parameter
                            normalized_grad = normalized_layer_grad[start_idx:start_idx + grad_size].view_as(p.grad.data)
                            start_idx += grad_size
                            
                            # Apply weight decay: g' = g' + λ*w
                            # This regularizes the model by penalizing large weights
                            if weight_decay:
                                normalized_grad = normalized_grad + weight_decay * p.data
                            
                            # Get optimizer state
                            state = self.state[p]
                            
                            # Apply momentum update: v_t = β*v_{t-1} + g'
                            buf = momentum_update(state, normalized_grad, momentum)
                            
                            # Handle adaptive scaling if enabled
                            if adaptive:
                                # Keep running sum of squared gradients: Σ v_t^2
                                if 'sum_sq_grad' not in state:
                                    state['sum_sq_grad'] = buf.detach().pow(2)
                                else:
                                    state['sum_sq_grad'].add_(buf.detach().pow(2))
                                # Scale by inverse sqrt of sum: v'_t = v_t / sqrt(Σ v_t^2 + ε)
                                # Similar to RMSprop/Adam's adaptive scaling
                                adaptive_buf = buf / (state['sum_sq_grad'].sqrt() + adaptive_eps)
                            else:
                                adaptive_buf = buf
                            
                            # Handle adaptive learning rate (inspired by Adagrad)
                            if adaptive_lr_flag:
                                # Maintain sum of squared gradients for learning rate adaptation
                                if 'sum_sq_lr' not in state:
                                    state['sum_sq_lr'] = normalized_grad.detach().pow(2)
                                else:
                                    state['sum_sq_lr'].add_(normalized_grad.detach().pow(2))
                                # Scale learning rate inversely to the sqrt of accumulated squared gradients
                                # lr_scale = 1/sqrt(Σ g'^2)
                                # This reduces lr for frequently updated parameters
                                lr_scale = 1.0 / (state['sum_sq_lr'].sqrt().mean() + adaptive_eps)
                                effective_lr = lr * lr_scale
                            else:
                                effective_lr = lr
                            
                            # Update parameter: w_t = w_{t-1} - η*v'_t
                            p.data.add_(adaptive_buf, alpha=-effective_lr)
            else:
                # Original group-based normalization logic with fixed group sizes
                for p in group['params']:
                    if p.grad is None:
                        continue
                    
                    # Apply gradient clipping if specified
                    if clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(p, clip_grad_norm)
                    
                    grad = p.grad.data
                    # Convert n-dimensional gradient tensor into 1D vector
                    flat_grad = grad.view(-1)
                    N = flat_grad.numel()
                    
                    # Determine group_size using provided values or dynamic selection
                    given_group_size = group.get('group_size')
                    given_num_groups = group.get('num_groups')
                    if given_group_size is not None:
                        group_size = given_group_size
                    elif given_num_groups is not None:
                        group_size = math.ceil(N / given_num_groups)
                    else:
                        # If neither is provided, calculate group_size dynamically:
                        # - Set number of groups to sqrt(N)
                        # - Then calculate group_size from that
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
                    # g' = g' + λ*w where λ is weight_decay
                    if weight_decay:
                        normalized_grad = normalized_grad + weight_decay * p.data
                    
                    # Get optimizer state
                    state = self.state[p]
                    
                    # Apply momentum update
                    buf = momentum_update(state, normalized_grad, momentum)
                    
                    # Handle adaptive scaling
                    if adaptive:
                        if 'sum_sq_grad' not in state:
                            state['sum_sq_grad'] = buf.detach().pow(2)
                        else:
                            state['sum_sq_grad'].add_(buf.detach().pow(2))
                        # Scale by inverse sqrt of sum: v'_t = v_t / sqrt(Σ v_t^2 + ε)
                        adaptive_buf = buf / (state['sum_sq_grad'].sqrt() + adaptive_eps)
                    else:
                        adaptive_buf = buf
                    
                    # Handle adaptive learning rate
                    if adaptive_lr_flag:
                        if 'sum_sq_lr' not in state:
                            state['sum_sq_lr'] = normalized_grad.detach().pow(2)
                        else:
                            state['sum_sq_lr'].add_(normalized_grad.detach().pow(2))
                        # Scale lr inversely to the square root of accumulated squared gradients
                        lr_scale = 1.0 / (state['sum_sq_lr'].sqrt().mean() + adaptive_eps)
                        effective_lr = lr * lr_scale
                    else:
                        effective_lr = lr
                    
                    # Update parameter: w_t = w_{t-1} - η*v'_t
                    p.data.add_(adaptive_buf, alpha=-effective_lr)
                    
        return loss
