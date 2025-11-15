"""
Sharpness-Aware Minimization (SAM) Optimizer

Reference: https://arxiv.org/abs/2010.01412
SAM improves model generalization by seeking parameters that lie in 
neighborhoods having uniformly low loss.
"""

import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        """
        Args:
            params: Model parameters
            base_optimizer: Base optimizer class (e.g., torch.optim.SGD, torch.optim.AdamW)
            rho: Neighborhood size
            adaptive: Use adaptive version (ASAM)
            **kwargs: Arguments for base optimizer
        """
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        First step: compute and apply gradient ascent in the direction of sharpness
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # Store original parameter
                self.state[p]["old_p"] = p.data.clone()
                
                # Compute epsilon
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                
                # Move parameter in the direction of gradient
                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Second step: compute gradient at perturbed point and restore original parameters
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # Restore original parameter
                p.data = self.state[p]["old_p"]

        # Update parameters using base optimizer
        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """
        Single step combining both first and second steps
        NOTE: closure must be provided when using this method
        """
        assert closure is not None, "SAM requires closure, but it was not provided"
        
        # Enable gradient computation
        closure = torch.enable_grad()(closure)

        # First forward-backward pass (perturb parameters)
        self.first_step(zero_grad=True)
        
        # Second forward-backward pass (compute gradient at perturbed point)
        with torch.enable_grad():
            closure()
        
        # Update using base optimizer
        self.second_step()

    def _grad_norm(self):
        """
        Compute the norm of gradients
        """
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


# Usage example in training loop:
"""
# Initialize
base_optimizer = torch.optim.AdamW
optimizer = SAM(model.parameters(), base_optimizer, lr=0.001, weight_decay=0.0001)

# Training loop
for batch in dataloader:
    # First forward-backward pass
    loss = loss_fn(model(inputs), targets)
    loss.backward()
    optimizer.first_step(zero_grad=True)
    
    # Second forward-backward pass
    loss_fn(model(inputs), targets).backward()
    optimizer.second_step(zero_grad=True)
"""