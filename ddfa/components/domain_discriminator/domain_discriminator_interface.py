'''
Implementation: Pranav Mani, Manley Roberts
'''

import random
import numpy as np
import torch

class DomainDiscriminator:

    def deterministic_seed(self, seed):
        # https://github.com/pytorch/pytorch/issues/11278
        random.seed(self.init_seed)
        np.random.seed(self.init_seed)
        torch.manual_seed(self.init_seed)
        torch.cuda.manual_seed_all(self.init_seed)
        torch.backends.cudnn.deterministic = True

    def fit_discriminator(self, train_data, valid_data, train_domains, valid_domains):
        pass

    def get_features(self, data):
        pass

    def get_hyperparameter_dict(self):
        pass
