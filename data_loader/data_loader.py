import torch
from torch.utils.data import DataLoader

class BasicDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers, sampler):

        use_cuda = torch.cuda.is_available()
        
        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'sampler': sampler,
            'pin_memory': use_cuda
        }
        super().__init__(**self.init_kwargs)