import os
import torch
import torchvision.datasets as datasets

class SUN397:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16):
        # Data loading code
        traindir = os.path.join(location, 'SUN397', 'train')
        valdir = os.path.join(location, 'SUN397', 'test')   #CHRISTOS COMMENT, SUN changed to capital to match dataset and 'val' changed to 'test'
        #ALSO SUN397 IS A PARTITION OF THE ACTUAL SUN397 WITH SAME AMOUNT OF DATA IN TRAIN AND TEST. HOW DID YOU DECIDE ON WHICH PARTITION TO TAKE?

        self.train_dataset = datasets.ImageFolder(traindir, transform=preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = datasets.ImageFolder(valdir, transform=preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
        idx_to_class = dict((v, k)
                            for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i][2:].replace('_', ' ') for i in range(len(idx_to_class))]
