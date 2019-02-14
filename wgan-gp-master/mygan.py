# -*- coding: utf-8 -*-
import torch
import os
import imageio
from torch.utils.data import Dataset, DataLoader
from models import Generator, Discriminator
from training import Trainer
from dataloaders import get_mnist_dataloaders
from torchvision import transforms, utils

img_size = (256, 256, 3)
# img_size = (32, 32, 1)
generator = Generator(img_size=img_size, latent_dim=100, dim=16)
discriminator = Discriminator(img_size=img_size, dim=16)

lr = 1e-4
betas = (.9, .99)
# Initialize optimizers
G_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)
D_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=betas)



nature_dataset = NatureDataset('dataset', transform=transforms.Compose([ToTensor()]))

#nature_dataset.get_sample_data()

data_loader = DataLoader(nature_dataset, batch_size=256, shuffle=True, num_workers=4)

#data_loader, _ = get_mnist_dataloaders(batch_size=512)

# Set up trainer
trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer,
                  use_cuda=torch.cuda.is_available())

# Train model for X epochs
trainer.train(data_loader, epochs=100, save_training_gif=True)


class NatureDataset(Dataset):
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root_dir))
    
    def __getitem__(self, idx):
        print("getting item from dataset")
        img_name = os.path.join(self.root_dir,f'img_ ({idx}).jpg')
        image = imageio.imread(img_name)
        sample = image
        
        if self.transform:
            sample = self.transform(sample)
        
        return {'image':sample, 'label':1}
    
    def get_sample_data(self):
        import matplotlib.pyplot as plt
        for i in range(len(nature_dataset)):
            sample=nature_dataset[i]
            
            ax = plt.subplot(1, 4, i + 1)
            plt.tight_layout()
            ax.set_title('Sample #{}'.format(i))
            ax.axis('off')
            plt.imshow(sample)
            
            if i == 3:
                plt.show()
                break
            
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(label)}
            


    
    
