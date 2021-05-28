import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils
from PIL import Image
import utilities.CelebASettings as settings


img_shape = settings.img_shape
img_channels_shape = settings.img_channels_shape

# transforms
tfms_train = transforms.Compose([
    transforms.Grayscale(),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(img_shape),
    transforms.ToTensor(),
    transforms.Normalize((0.3654104799032211,), (0.06553166485374898,))
])

tfms_val = tfms_test = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(img_shape),
    transforms.ToTensor(),
    transforms.Normalize((0.3654104799032211,), (0.06553166485374898,))
])

class ImageTripletDataset(Dataset):
    """Dataset of triplets of images"""

    def __init__(self, ImageDataset, num_triplets, criteria):
        """
        Args:
            ImageDataset (torch.utils.data.Dataset):  A pytorch dataset that serves individual images
            num_triplets (int): Number of triplets
        """
        self.imagedataset=ImageDataset
        self.num_triplets = num_triplets
        self.criteria = criteria

        # generate indices list
        self.indices = np.random.randint(0, len(self.imagedataset), (self.num_triplets, 3))
        # TODO: make sure there are no duplicates?

    def __len__(self):
        return self.num_triplets

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        Aix, Bix, Cix = self.indices[idx]

        A = self.imagedataset[Aix]
        B = self.imagedataset[Bix]
        C = self.imagedataset[Cix]

        A_img, A_attr = A['image'], A['attributes']
        B_img, B_attr = B['image'], B['attributes']
        C_img, C_attr = C['image'], C['attributes']

        sample = {'A': A_img, 'B': B_img, 'C':C_img, 
                  'target': self.criteria(A_attr,B_attr,C_attr),
                  'image_indices': (Aix, Bix, Cix), 'image_digits': (A_attr, B_attr, C_attr)}

        return sample
    

class CelebA(Dataset):
  def __init__(self, root_dir: str, partition: str, transform=None):
    # the image data
    self.img_dir = f"{root_dir}/img_align_celeba"

    # the partition data
    if partition=='train':
      self.partition_ix = 0
    elif partition=='val':
      self.partition_ix = 1
    elif partition=='test':
      self.partition_ix = 2
    else:
      raise ValueError("partition must be one of 'train', 'val', or 'test'")

    df_partitions = pd.read_csv(f"{root_dir}/list_eval_partition.csv")
    self.df_partitions = df_partitions[df_partitions['partition']==self.partition_ix]

    # the attribute data
    df_attributes = pd.read_csv(f"{root_dir}/list_attr_celeba.csv")

    # filter the attributes
    self.df_attributes = self.df_partitions.merge(df_attributes, how='left', on='image_id').drop(columns=['partition'])

    # transforms
    self.transform = transform

  def __len__(self):
    return self.df_attributes.shape[0]

  def __getitem__(self, ix):
    # image
    filepath = f"{self.img_dir}/{self.df_attributes.image_id[ix]}"
    image = Image.open(filepath)
    if self.transform:
      image = self.transform(image)

    # attributes
    attributes =self.df_attributes.iloc[ix, 1:].to_dict()

    return {'image': image, 'attributes': attributes}

def make_data(batch_size, batch_size_test, kwargs, n_train, n_test, metacriteria, data_dir):
    image_dataset_train = CelebA(data_dir, "train", transform=tfms_train)
    triplet_dataset_train = ImageTripletDataset(image_dataset_train, n_train, metacriteria)
    train_loader = torch.utils.data.DataLoader(
      triplet_dataset_train,
      batch_size=batch_size, shuffle=True, **kwargs)

    # testing data
    image_dataset_test = CelebA(data_dir, "test", transform=tfms_test)
    triplet_dataset_test = ImageTripletDataset(image_dataset_test, n_test, metacriteria)
    test_loader = torch.utils.data.DataLoader(
      triplet_dataset_test,
      batch_size=batch_size_test, shuffle=True, **kwargs)

    return train_loader,test_loader