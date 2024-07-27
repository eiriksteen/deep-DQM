import torch


class RefBuilder():

    def __init__(self, num_features: int, num_bins: int, device: str = "cpu"):

        self.num_features = num_features
        self.num_bins = num_bins
        self.device = device

        self.mu = torch.zeros(num_features, num_bins).to(device)
        self.var = torch.zeros(num_features, num_bins).to(device)

    def compute_reference(
            self,
            x_neg: torch.Tensor,
            alpha: torch.Tensor | float
    ):

        mu, var = self.mu, self.var

        for i, x_ in enumerate(x_neg):

            alpha_ = alpha[i] if isinstance(alpha, torch.Tensor) else alpha
            mu = (1 - alpha_) * mu + alpha_ * x_
            var = (1 - alpha_) * var + alpha_ * (x_ - mu)**2

        return mu, var

    def update_reference(
            self,
            x_neg: torch.Tensor,
            alpha: torch.Tensor):

        mu, var = self.compute_reference(x_neg, alpha)

        self.mu = mu
        self.var = var
