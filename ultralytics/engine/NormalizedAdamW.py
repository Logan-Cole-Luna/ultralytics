# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import math
from typing import cast, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT

__all__ = ["NormalizedAdamW"]


class NormalizedAdamW(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[Union[float, Tensor], Union[float, Tensor]] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
        # Additional parameters for gradient normalization
        normalize: bool = True,
        layer_wise: bool = True,
        group_size: Optional[int] = None,
        scale_aware: bool = False,
        scale_factor: float = 0.2,
        max_group_size: int = 5000,
        clip_norm: Optional[float] = None,
    ):
        """
        Implements Normalized AdamW algorithm with added gradient normalization.
        
        This optimizer extends AdamW by normalizing gradients within groups of parameters,
        helping to balance the scale of updates across different layers of the network.
        
        Args:
            params (iterable): Iterable of parameters to optimize or dicts defining parameter groups
            lr (float, Tensor): Learning rate
            betas (tuple[float, float]): Coefficients used for computing running averages of gradient
                and its square (default: (0.9, 0.999))
            eps (float): Term added to the denominator to improve numerical stability
            weight_decay (float): Weight decay (L2 penalty)
            amsgrad (bool): Whether to use the AMSGrad variant
            maximize (bool): Maximize the params based on the objective, instead of minimizing
            foreach (bool, optional): Whether to use foreach implementation
            capturable (bool): Whether to use capturable implementation
            differentiable (bool): Whether to create a differentiable optimizer
            fused (bool, optional): Whether to use fused implementation if available
            normalize (bool): Enable gradient normalization
            layer_wise (bool): Group parameters by layer rather than fixed size
            group_size (int, optional): Fixed number of parameters per group when layer_wise=False
            scale_aware (bool): Preserve some gradient scale information
            scale_factor (float): Mix factor for scale-aware normalization
            max_group_size (int): Maximum parameters per sub-group
            clip_norm (float, optional): Clip gradient norm value
        """
        if isinstance(lr, Tensor):
            if foreach and not capturable:
                raise ValueError("lr as a Tensor is not supported for capturable=False and foreach=True")
            if lr.numel() != 1:
                raise ValueError("Tensor lr must be 1-element")

        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not (
            (isinstance(betas[0], float) and isinstance(betas[1], float))
            or (isinstance(betas[0], Tensor) and isinstance(betas[1], Tensor))
        ):
            raise ValueError("betas must be either both floats or both Tensors")
        if isinstance(betas[0], Tensor):
            if not capturable and foreach:
                raise ValueError("betas[0] as a Tensor is not supported for capturable=False and foreach=True")
            if betas[0].numel() != 1:
                raise ValueError("Tensor betas[0] must be 1-element")
        if isinstance(betas[1], Tensor):
            if not capturable and foreach:
                raise ValueError("betas[1] as a Tensor is not supported for capturable=False and foreach=True")
            if betas[1].numel() != 1:
                raise ValueError("Tensor betas[1] must be 1-element")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            foreach=foreach,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
            # Normalization parameters
            normalize=normalize,
            layer_wise=layer_wise,
            group_size=group_size,
            scale_aware=scale_aware,
            scale_factor=scale_factor,
            max_group_size=max_group_size,
            clip_norm=clip_norm,
        )
        super().__init__(params, defaults)

        if fused:
            if differentiable:
                raise RuntimeError("`fused` does not support `differentiable`")
            self._step_supports_amp_scaling = True
            if foreach:
                raise RuntimeError("`fused` and `foreach` cannot be `True` together.")
                
        # Create parameter name mapping for layer-wise grouping
        self.param_to_layer = {}
        
        # If layer_wise is enabled, organize parameters by layer
        if layer_wise and normalize:
            self._organize_layer_groups()

    def _organize_layer_groups(self):
        """
        Organize parameters into layer groups based on parameter names.
        
        This enables layer-wise normalization where gradients from the same
        logical layer are normalized together, preserving intra-layer relationships.
        """
        # Build unique layer names from parameter groups
        self.param_to_layer = {}
        layer_names = set()
        
        # Try to extract parameter names from their full names
        for group_idx, group in enumerate(self.param_groups):
            for param_idx, param in enumerate(group['params']):
                # Use param identifier with group/param index as backup names
                param_id = f"group{group_idx}_param{param_idx}"
                
                # Try to find parameter name in state_dict keys
                for name, p in self._params_to_names(param, param_id).items():
                    if p is param:
                        # Extract the layer name (e.g., 'conv1.weight' -> 'conv1')
                        layer_name = name.split('.')[0] if '.' in name else name
                        layer_names.add(layer_name)
                        self.param_to_layer[param] = layer_name
        
        # Create mapping from layer name to index
        layer_indices = {name: idx for idx, name in enumerate(sorted(layer_names))}
        
        # Map each parameter to its layer index
        for param, layer_name in list(self.param_to_layer.items()):
            self.param_to_layer[param] = layer_indices.get(layer_name, 0)

    def _params_to_names(self, target_param, default_name):
        """Helper to find parameter names for layer-wise grouping."""
        # This is a simplified version that uses a default name scheme
        # In practice, you would scan the model's state_dict to find actual names
        return {default_name: target_param}

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            group.setdefault("fused", False)
            # Normalization defaults
            group.setdefault("normalize", True)
            group.setdefault("layer_wise", True)
            group.setdefault("group_size", None)
            group.setdefault("eps", 1e-5)
            group.setdefault("scale_aware", False)
            group.setdefault("scale_factor", 0.2)
            group.setdefault("max_group_size", 5000)
            group.setdefault("clip_norm", None)
            fused = group.setdefault("fused", None)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    p_state["step"] = (
                        torch.tensor(
                            step_val,
                            dtype=torch.float,
                            device=p.device,
                        )
                        if group["capturable"] or group["fused"]
                        else torch.tensor(step_val, dtype=torch.float)
                    )

    def _normalize_gradients(self, group, params, grads):
        """
        Apply normalization to gradients based on group settings.
        
        Args:
            group (dict): Parameter group containing settings
            params (list): List of parameters
            grads (list): List of gradients
        """
        if not group["normalize"] or not grads:
            return
            
        layer_wise = group["layer_wise"]
        eps = group["eps"]
        
        if layer_wise:
            # Group parameters by layer
            layer_params_dict = {}
            for param, grad in zip(params, grads):
                layer_idx = self.param_to_layer.get(param, 0)
                if layer_idx not in layer_params_dict:
                    layer_params_dict[layer_idx] = {"params": [], "grads": []}
                layer_params_dict[layer_idx]["params"].append(param)
                layer_params_dict[layer_idx]["grads"].append(grad)
            
            # Normalize each layer separately
            for layer_idx, layer_data in layer_params_dict.items():
                self._normalize_layer(group, layer_data["params"], layer_data["grads"])
        else:
            # Normalize each parameter individually with fixed group size
            for param, grad in zip(params, grads):
                self._normalize_fixed_size_groups(group, param, grad)
    
    def _normalize_layer(self, group, params, grads):
        """
        Normalize gradients within a layer group.
        
        Args:
            group (dict): Parameter group containing settings
            params (list): List of parameters in the layer
            grads (list): List of gradients in the layer
        """
        # Extract settings
        eps = group["eps"]
        max_group_size = group["max_group_size"]
        scale_aware = group["scale_aware"]
        scale_factor = group["scale_factor"]
        
        # Collect all gradients in this layer
        all_grads = []
        for grad in grads:
            if grad.is_sparse:
                # Skip sparse gradients for normalization
                continue
            all_grads.append(grad.view(-1))
        
        if not all_grads:
            return
            
        # Concatenate all gradients from this layer
        layer_grad = torch.cat(all_grads)
        
        # Split large layers into sub-groups if needed
        if layer_grad.numel() > max_group_size:
            # Calculate actual sub-group size to ensure even division
            num_subgroups = (layer_grad.numel() + max_group_size - 1) // max_group_size
            # Make sure we can evenly divide by adjusting the group size
            actual_group_size = (layer_grad.numel() + num_subgroups - 1) // num_subgroups
            
            # Process each subgroup separately without reshaping into a matrix
            normalized_layer_grad = torch.zeros_like(layer_grad)
            for i in range(num_subgroups):
                start_idx = i * actual_group_size
                end_idx = min(start_idx + actual_group_size, layer_grad.numel())
                if start_idx >= layer_grad.numel():
                    break
                    
                # Get this subgroup
                subgroup = layer_grad[start_idx:end_idx]
                
                # Normalize this subgroup
                sub_mean = subgroup.mean()
                sub_std = subgroup.std() + eps
                
                if scale_aware:
                    # Mix original gradients with normalized ones for scale awareness
                    normalized_subgroup = scale_factor * subgroup + (1 - scale_factor) * (
                        (subgroup - sub_mean) / sub_std)
                else:
                    normalized_subgroup = (subgroup - sub_mean) / sub_std
                    
                # Store back in the full gradient tensor
                normalized_layer_grad[start_idx:end_idx] = normalized_subgroup
        else:
            # Standard normalization for smaller layers
            layer_mean = layer_grad.mean()
            layer_std = layer_grad.std() + eps
            
            if scale_aware:
                normalized_layer_grad = scale_factor * layer_grad + (1 - scale_factor) * (
                    (layer_grad - layer_mean) / layer_std)
            else:
                normalized_layer_grad = (layer_grad - layer_mean) / layer_std
        
        # Distribute normalized gradients back to parameters
        start_idx = 0
        grad_idx = 0
        for i, grad in enumerate(grads):
            if grad.is_sparse:
                # Skip sparse gradients
                continue
                
            grad_size = grad.numel()
            # Extract the portion of normalized gradients for this parameter
            normalized_grad = normalized_layer_grad[start_idx:start_idx + grad_size].view_as(grad)
            grad.copy_(normalized_grad)  # In-place update of gradient
            start_idx += grad_size
            grad_idx += 1
    
    def _normalize_fixed_size_groups(self, group, param, grad):
        """
        Normalize gradients using fixed-size groups for a single parameter.
        
        Args:
            group (dict): Parameter group containing settings
            param (Tensor): Parameter tensor
            grad (Tensor): Gradient tensor
        """
        if grad.is_sparse:
            # Skip sparse gradients
            return
            
        eps = group["eps"]
        given_group_size = group["group_size"]
        scale_aware = group["scale_aware"]
        scale_factor = group["scale_factor"]
        
        flat_grad = grad.view(-1)
        N = flat_grad.numel()
        
        # Determine group_size
        if given_group_size is not None:
            group_size = given_group_size
        else:
            # Default to sqrt(N) groups
            dynamic_num_groups = max(1, int(math.sqrt(N)))
            group_size = math.ceil(N / dynamic_num_groups)
        
        # Skip normalization for very small gradients
        if group_size < 2 or N < 2:
            return
            
        remainder = N % group_size
        
        # If N is not perfectly divisible by group_size, pad with zeros
        if remainder != 0:
            pad_size = group_size - remainder
            flat_grad_padded = torch.cat([flat_grad, flat_grad.new_zeros(pad_size)], dim=0)
        else:
            flat_grad_padded = flat_grad
            
        # Reshape into matrix of shape (⌈N/group_size⌉, group_size)
        reshaped = flat_grad_padded.view(-1, group_size)
        
        # Calculate mean and standard deviation for each group
        group_mean = reshaped.mean(dim=1, keepdim=True)
        group_std = reshaped.std(dim=1, keepdim=True) + eps
        
        # Normalize each group
        if scale_aware:
            normalized = scale_factor * reshaped + (1 - scale_factor) * (
                (reshaped - group_mean) / group_std)
        else:
            normalized = (reshaped - group_mean) / group_std
        
        # Convert back to 1D and remove padding
        normalized_flat_grad = normalized.view(-1)[:N]
        
        # Update gradient in-place
        grad.copy_(normalized_flat_grad.view_as(grad))
    
    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        amsgrad,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
    ):
        has_complex = False
        for p in group["params"]:
            if p.grad is None:
                continue
                
            # Apply gradient clipping if specified
            if group["clip_norm"] is not None:
                torch.nn.utils.clip_grad_norm_(p, group["clip_norm"])
                
            has_complex |= torch.is_complex(p)
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError("NormalizedAdamW does not support sparse gradients")
            grads.append(p.grad)

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                if group["fused"]:
                    # No need to type-check for fused here as we already did above
                    pass
                # Note: deliberately host step on CPU if capturable & fused are off
                # This is because kernel launches are costly on CUDA and XLA
                state["step"] = (
                    torch.zeros((), dtype=torch.float, device=p.device)
                    if group["capturable"] or group["fused"]
                    else torch.tensor(0.0, dtype=torch.float)
                )
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])

            if group["amsgrad"]:
                max_exp_avg_sqs.append(state["max_exp_avg_sq"])
            if group["differentiable"] and state["step"].requires_grad:
                raise RuntimeError(
                    "`requires_grad` is not supported for `step` in differentiable mode"
                )

            # Foreach without capturable does not support a tensor lr
            if (
                group["foreach"]
                and isinstance(group["lr"], Tensor)
                and not group["capturable"]
            ):
                raise RuntimeError(
                    "lr as a Tensor is not supported for capturable=False and foreach=True"
                )

            state_steps.append(state["step"])
        return has_complex

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group["amsgrad"]
            beta1, beta2 = group["betas"]

            # Initialize groups and collect parameters, gradients, states
            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                amsgrad,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )

            # First, apply normalization to gradients if enabled
            if group["normalize"]:
                self._normalize_gradients(group, params_with_grad, grads)

            # Now apply AdamW update
            self._normalized_adamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=amsgrad,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                fused=group["fused"],
                grad_scale=None,
                found_inf=None,
                has_complex=has_complex,
            )

        return loss

    def _normalized_adamw(
        self,
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        max_exp_avg_sqs: List[Tensor],
        state_steps: List[Tensor],
        foreach: Optional[bool] = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
        grad_scale: Optional[Tensor] = None,
        found_inf: Optional[Tensor] = None,
        has_complex: bool = False,
        *,
        amsgrad: bool,
        beta1: float,
        beta2: float,
        lr: Union[float, Tensor],
        weight_decay: float,
        eps: float,
        maximize: bool,
    ):
        """
        Functional API for normalized AdamW algorithm computation.
        This extends torch.optim._functional.adamw with our normalization logic.
        """
        # Handle empty params list
        if len(params) == 0:
            return

        # Use single tensor implementation by default
        if foreach:
            func = self._multi_tensor_adamw
        elif fused:
            func = self._fused_adamw
        else:
            func = self._single_tensor_adamw

        func(
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
            amsgrad=amsgrad,
            beta1=beta1,
            beta2=beta2,
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
            grad_scale=grad_scale,
            found_inf=found_inf,
            has_complex=has_complex,
        )
    
    def _single_tensor_adamw(
        self,
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        max_exp_avg_sqs: List[Tensor],
        state_steps: List[Tensor],
        grad_scale: Optional[Tensor],
        found_inf: Optional[Tensor],
        *,
        amsgrad: bool,
        beta1: Union[Tensor, float],
        beta2: Union[Tensor, float],
        lr: Union[Tensor, float],
        weight_decay: float,
        eps: float,
        maximize: bool,
        capturable: bool,
        differentiable: bool,
        has_complex: bool,
    ):
        """Single tensor implementation of AdamW optimizer"""
        assert grad_scale is None and found_inf is None

        for i, param in enumerate(params):
            grad = grads[i] if not maximize else -grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step_t = state_steps[i]

            # Ensures compatibility with capturable tensor
            if capturable:
                assert param.device.type == step_t.device.type, \
                    f"If capturable=True, params and state_steps must be on same device type."

            # Handle complex parameters
            if torch.is_complex(param):
                grad = torch.view_as_real(grad)
                exp_avg = torch.view_as_real(exp_avg)
                exp_avg_sq = torch.view_as_real(exp_avg_sq)
                if amsgrad:
                    max_exp_avg_sqs[i] = torch.view_as_real(max_exp_avg_sqs[i])
                param_data = torch.view_as_real(param.data)
            else:
                param_data = param.data

            # Update step
            step_t += 1

            # Perform stepweight decay - use .data to avoid in-place operation error
            param_data.mul_(1 - lr * weight_decay)

            # Get beta1 on correct device
            if isinstance(beta1, Tensor):
                beta1_t = beta1.to(device=param.device, dtype=param.dtype)
            else:
                beta1_t = beta1

            # Decay the first and second moment running average coefficient
            exp_avg.lerp_(grad, 1 - beta1_t)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            # Extra steps for capturable or differentiable modes
            if capturable or differentiable:
                step = step_t

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                step_size = lr / bias_correction1
                step_size_neg = step_size.neg()

                bias_correction2_sqrt = bias_correction2.sqrt()

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    if differentiable:
                        max_exp_avg_sq = max_exp_avg_sqs[i].clone()
                    else:
                        max_exp_avg_sq = max_exp_avg_sqs[i]

                    max_exp_avg_sqs[i].copy_(torch.maximum(max_exp_avg_sq, exp_avg_sq))

                    # Uses the max. for normalizing running avg. of gradient
                    # Folds in step_size math here to avoid extra param-set-sized read+write
                    denom = (
                        max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt * step_size_neg)
                    ).add_(eps / step_size_neg)
                else:
                    denom = (
                        exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)
                    ).add_(eps / step_size_neg)

                # Use .data to avoid in-place operation error
                param_data.addcdiv_(exp_avg, denom)
            else:
                step = step_t.item()

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                step_size = lr / bias_correction1

                bias_correction2_sqrt = bias_correction2 ** 0.5

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])

                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
                else:
                    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

                # Use .data to avoid in-place operation error
                param_data.addcdiv_(exp_avg, denom, value=-step_size)

            # Restore complex view if needed
            if amsgrad and torch.is_complex(params[i]):
                max_exp_avg_sqs[i] = torch.view_as_complex(max_exp_avg_sqs[i])
    
    # These methods are placeholders for the multi-tensor and fused implementations
    # In a real implementation, we would adapt the original PyTorch code for these cases
    def _multi_tensor_adamw(self, *args, **kwargs):
        """Multi tensor implementation of AdamW optimizer (uses foreach APIs)"""
        # For a complete implementation, adapt the PyTorch _multi_tensor_adamw code here
        # This is a simplified placeholder that falls back to the single tensor implementation
        return self._single_tensor_adamw(*args, **kwargs)
    
    def _fused_adamw(self, *args, **kwargs):
        """Fused implementation of AdamW optimizer"""
        # For a complete implementation, adapt the PyTorch _fused_adamw code here
        # This is a simplified placeholder that falls back to the single tensor implementation
        return self._single_tensor_adamw(*args, **kwargs)
