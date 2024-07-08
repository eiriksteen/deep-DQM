import torch


class ReplayBuffer:
    def __init__(self, in_dim, buffer_size):
        assert buffer_size % 2 == 0, "positive and negative labeled buffers should be equal size"
        self.positive_buffer = torch.empty(buffer_size//2, in_dim)
        self.negative_buffer = torch.empty(buffer_size//2, in_dim)

    def __call__(self, batch):
        current_param = self.param[0]
        # Your collate logic here using batch and current_param
        pass

    def update_param(self, new_value):
        self.param[0] = new_value
