#TODOs:
# hudson - add saving of reconstruction images at the end of each epoch
# hudson - move model code into imported script
# hudson - add argument for choosing between vgg and bce reconstruction loss
# hudson - figure out why 1 channel output isn't breaking vgg loss
# nate - rework autoencoder to closely match tensorflow code
# nate/will - write bash script to run series of experiments for different hyperparameter choices
# nate/will - reorganize source folder!


from collections import OrderedDict
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.model_selection import train_test_split

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb
wandb.login()
wandb_logger = WandbLogger(project='face-morphology', log_model=True)

from utils import dataset_with_indices
from VGG_Loss import VGG_Loss

#---------------
# Main VAE Class
#---------------
class VAE(pl.LightningModule):
    def __init__(self, image_shape, n_mid, n_res, z_dim, data_size, val_frac, batch_size, num_workers, beta, vgg, fast_dev_run, lr):
        super(VAE, self).__init__()
        
        #self.hparams=hparams        
        #following are assingment of the constructor arguments to the class object
        self.image_shape=image_shape
        self.n_mid=n_mid
        self.n_res=n_res
        self.z_dim=z_dim
        self.data_size=data_size
        self.val_frac=val_frac
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.beta=beta
        self.vgg=vgg
        self.fast_dev_run=fast_dev_run
        self.lr=lr
        
        # load the vgg loss and freeze parameters
        self.vgg_loss = VGG_Loss().to(self.device).eval()
        for pars in self.vgg_loss.parameters():
            pars.requires_grad = False
        
        # encoder and decoder
        self.encoder = Encoder(input_shape = self.image_shape, 
                               n_mid = self.n_mid, 
                               n_res = self.n_res, 
                               kernel_size=3)
        self.decoder = Decoder(output_shape=self.image_shape, 
                               n_mid = self.n_mid, 
                               n_res = self.n_res, 
                               kernel_size=5)
        
        # linear transforms
        shape = self.image_shape
        self.h_dim = (shape[0]*shape[1]*shape[2]) // (4**4) * self.n_mid[-1]
        self.lin_mu = nn.Linear(self.h_dim, self.z_dim)
        self.lin_logvar = nn.Linear(self.h_dim, self.z_dim)
        self.lin_decode = nn.Linear(self.z_dim, self.h_dim)
        
        # dataset
        self.tfms = transforms.Compose([
            transforms.Grayscale(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((shape[1], shape[2])),
            transforms.ToTensor()
        ])
        
        dataClass = dataset_with_indices(datasets.ImageFolder)
        self.dataset = dataClass(root='../../data/datasets/celeba/',
                                 transform=self.tfms)
        
        # Creating subsets of the data for training and testing
        if self.data_size:
            indices =  torch.randperm(len(self.dataset))[:self.data_size]
        else:
            indices = np.arange(len(self.dataset))
        train_indices, val_indices = train_test_split(indices, test_size=self.val_frac)
        self.train_dataset = Subset(self.dataset, train_indices)
        self.val_dataset = Subset(self.dataset, val_indices)
        
        # set aside some validation images for 
        num_images = 10
        self.img_dataset = Subset(self.dataset, val_indices[:num_images])
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, 
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, 
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           pin_memory=True)
    
    def img_dataloader(self):
        return torch.utils.data.DataLoader(self.img_dataset,
                                           batch_size=len(self.img_dataset),
                                           num_workers=self.num_workers,
                                           pin_memory=True)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn_like(std)
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.lin_mu(h), self.lin_logvar(h)
        
        if self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        return z, mu, logvar
        
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]
    
    def forward(self, x):
        h = self.encoder(x)
        
        z, mu, logvar = self.bottleneck(h)
        z = self.lin_decode(z)
        
        return self.decoder(z), mu, logvar 

    def loss(self, x_hat, x, mu, logvar):
        
        # Reconstruction loss:
        if self.vgg:
            vgg = self.vgg_loss(x_hat, x)
            recon = vgg
            with torch.no_grad():
                bce = F.binary_cross_entropy(x_hat, x)
        else: 
            bce = F.binary_cross_entropy(x_hat, x)
            recon = bce
            with torch.no_grad():
                vgg = self.vgg_loss(x_hat, x)
        
        # KLD loss:
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        

        
        return recon + self.beta * kld, bce, kld, vgg
    
    def training_step(self, batch, batch_idx):
        x, _, _ = batch
        
        x_hat, mu, logvar = self.forward(x)
        loss, bce, kld, vgg = self.loss(x_hat, x, mu, logvar)

        tensorboard_logs = {'train_loss': loss, 
                            'train_bce': bce, 
                            'train_kld': kld,
                            'train_vgg': vgg}

        for key, val in tensorboard_logs.items():
            self.log(key, val)#, sync_dist=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, _, _ = batch
        
        if self.fast_dev_run:
            print(x.shape)
        
        x_hat, mu, logvar = self.forward(x)
        loss, bce, kld, vgg = self.loss(x_hat, x, mu, logvar)

        tensorboard_logs = {'val_loss': loss,
                            'val_bce': bce,
                            'val_kld': kld,
                            'val_vgg': vgg}
        for key, val in tensorboard_logs.items():
            self.log(key, val)#, sync_dist=True)

        return {'val_loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        
        return {'val_loss': avg_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def training_epoch_end(self, outputs):
        img_loader = self.img_dataloader()
        
        for ix, (x, _, _) in enumerate(img_loader):
            if ix==0:
                x_hat, _, _ = self.forward(x.to(self.device))
                stack = torch.stack((x, x_hat), dim=1).view(2*x.shape[0], x.shape[1], x.shape[2], x.shape[3])
                grid = make_grid(stack, nrow=2)[0][...,None].detach().numpy()
                self.log('reconstructions',
                         wandb.Image(grid, caption="Face Reconstructions"))

    
    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('--lr', default=1e-2, type=float)
        parser.add_argument('--batch_size', default=128, type=int)
        parser.add_argument('--image_shape', default=[1, 64, 64], type=int, nargs=3)
        parser.add_argument('--data_size', default=None, type=int, help="Used to subsample dataset for quick testing.")
        parser.add_argument('--val_frac', default=0.2, type=float)
        parser.add_argument('--n_mid', default=[96, 64, 32, 32], type=int, nargs=4)
        parser.add_argument('--n_res', default=32, type=int)
        parser.add_argument('--z_dim', default=128, type=int)
        parser.add_argument('--beta', default=0.01, type=float)
        parser.add_argument('--vgg', default=False, type=str, help="Use VGG rather than BCE reconstruction loss"),
        parser.add_argument('--num_workers', default=4, type=int)
        parser.add_argument('--fast_dev_run', default=False, type=bool)
        return parser

    
#---------------
# Helper Classes
#---------------
class ResNetBlock(nn.Module):
    r"""Resnet style block for predicting color shift masks on input images. 
    
    Args:
        num_in (int) - number of input channels (and output channels)
        num_features (int) - number of intermediate channels in resnet block
    """

    def __init__(self, num_in, num_mid, kernel_size=5):

        super(ResNetBlock, self).__init__()
        
        self.res = nn.Sequential(OrderedDict([
            # conv block 1
            ('conv0', nn.Conv2d(num_in, num_mid, kernel_size, stride=1,
                                padding=(kernel_size-1)//2, bias=False)),
            ('norm0', nn.BatchNorm2d(num_mid)),
            ('relu0', nn.ReLU(inplace=True)),
            # conv block 2
            ('conv1', nn.Conv2d(num_mid, num_in, kernel_size, stride=1,
                                padding=(kernel_size-1)//2, bias=False)),
            ('norm1', nn.BatchNorm2d(num_in)),

        ]))
        
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        # resnet style output: add input to features at relu
        return self.relu1(x + self.res(x))
    
class EncoderBlock(nn.Module):
    def __init__(self, n_in, n_mid, n_out, kernel_size=5):
        super(EncoderBlock, self).__init__()
        
        self.block = nn.Sequential(
            ResNetBlock(n_in, n_mid, kernel_size),
            nn.Conv2d(n_in, n_out, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=2),
            nn.BatchNorm2d(n_out),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.block(x)
        
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class Encoder(nn.Module):
    def __init__(self, input_shape, n_mid=(96,64,32,32), n_res=32, kernel_size=5):
        super(Encoder, self).__init__()
        
        self.encoder = nn.Sequential(
            # start
            nn.Conv2d(input_shape[0], n_mid[0], kernel_size=7, padding=3, stride=2),
            #nn.MaxPool2d(kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(n_mid[0]),
            nn.ReLU(),
            # encoding blocks
            EncoderBlock(n_mid[0], n_res, n_mid[1], kernel_size),
            EncoderBlock(n_mid[1], n_res, n_mid[2], kernel_size),
            EncoderBlock(n_mid[2], n_res, n_mid[3], kernel_size),
            Flatten()
        )
        
    def forward(self, x):
        return self.encoder(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, n_in, n_mid, n_out, kernel_size=6):
        super(DecoderBlock, self).__init__()
        
        self.block = nn.Sequential(
            ResNetBlock(n_in, n_mid, kernel_size),
            nn.Conv2d(n_in, n_out, kernel_size=kernel_size, padding=(kernel_size-1)//2),
            nn.BatchNorm2d(n_out),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear") #changed from nearest to bilinear 
        )
        
    def forward(self, x):
            return self.block(x)
        
class UnFlatten(nn.Module):
    def __init__(self, shape):
        super(UnFlatten, self).__init__()
        self.shape = shape
        
    def forward(self, input):
        return input.view(input.size(0), self.shape[0], self.shape[1], self.shape[2])
    
class Decoder(nn.Module):
    def __init__(self, output_shape=(1,64,64), n_mid=(96,64,32,32), n_res=32, kernel_size=7):
        super(Decoder, self).__init__()
          
        self.decoder = nn.Sequential(
            UnFlatten((n_mid[-1], output_shape[1]//2**4, output_shape[2]//2**4)),
            nn.Upsample(scale_factor=2, mode="bilinear"), #changed from nearest to bilinear 
            DecoderBlock(n_mid[-1], n_res, n_mid[-2], kernel_size),
            DecoderBlock(n_mid[-2], n_res, n_mid[-3], kernel_size),
            DecoderBlock(n_mid[-3], n_res, n_mid[-4], kernel_size),
            nn.Conv2d(n_mid[-4], output_shape[0], kernel_size, padding=(kernel_size-1)//2),
            nn.BatchNorm2d(output_shape[0]),
            nn.Sigmoid()
        )  
        
    def forward(self, x):
        x = self.decoder(x)
        
        return x

    
#---------------
# Main Function
#---------------
def main(hparams):
    #init values to pass into model
    image_shape = hparams.image_shape
    n_mid = hparams.n_mid
    n_res = hparams.n_res
    z_dim = hparams.z_dim
    data_size = hparams.data_size
    val_frac = hparams.val_frac
    batch_size = hparams.batch_size
    num_workers = hparams.num_workers
    beta = hparams.beta
    vgg = hparams.vgg
    fast_dev_run = hparams.fast_dev_run
    lr = hparams.lr
    # init module
    model = VAE(image_shape, n_mid, n_res, z_dim, data_size, val_frac, batch_size, num_workers, beta, vgg, fast_dev_run, lr)

    wandb_logger.watch(model, log='gradients')

    # most basic trainer, uses good defaults
    trainer = pl.Trainer(
        max_epochs=hparams.max_epochs,
        callbacks=[EarlyStopping(monitor='val_loss', patience=6),
                  ModelCheckpoint(monitor='val_loss')],
        auto_lr_find=hparams.lr_tune,
        accelerator='ddp',
        gpus=hparams.gpus,
        logger=[wandb_logger],
        profiler=hparams.profiler,
        fast_dev_run=hparams.fast_dev_run
    )
    trainer.tune(model)
    trainer.fit(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False, 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpus', type=str, default="0,1", help='list which gpu device(s) to use. For example 0,1,2,3')
    parser.add_argument('--lr_tune', default=False, type=bool)
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--profiler', default=None, type=str)

    # give the module a chance to add own params
    # good practice to define LightningModule speficic params in the module
    parser = VAE.add_model_specific_args(parser)

    # parse params
    hparams = parser.parse_args()
    pl.utilities.seed.seed_everything(hparams.seed)

    main(hparams)
