
import torch

class SubjectNormalizer:
    def __init__(self):
        self.stats = {}

    def fit(self, eegs, subject_idxs):
        subject_idxs = torch.tensor(subject_idxs)  
        for subj in torch.unique(subject_idxs):
            mask = subject_idxs == subj
            data = torch.tensor(eegs[mask])  
            mean = data.mean(dim=(0, 2), keepdim=True)
            std = data.std(dim=(0, 2), keepdim=True) + 1e-6
            self.stats[int(subj)] = (mean, std)

    def transform(self, eegs, subject_idxs):
        subject_idxs = torch.tensor(subject_idxs)  
        eegs = torch.tensor(eegs, dtype=torch.float32)  
        normed = torch.zeros_like(eegs)
        for i in range(eegs.shape[0]):
            mean, std = self.stats[int(subject_idxs[i])]
            normed[i] = (eegs[i] - mean) / std
        return normed
