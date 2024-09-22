import torch
from torch.utils.data import Dataset, DataLoader
import pandas
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torchvision


def save_images(images, file_name):
    torchvision.utils.save_image(images / 255, file_name)

def plot_tsne(dataset, latents, file_name, plot_title="t-SNE Plot"):
    lc = latents.detach().cpu().numpy()
    tsne = TSNE(n_components=2, random_state=42)
    lat_tsne = tsne.fit_transform(lc)
    plt.figure(figsize=(8, 6))
    labels = dataset.y
    class_names = {0: 'T-shirt',
                   1: 'Trouser',
                   2: 'Pullover',
                   3: 'Dress',
                   4: 'Coat',
                   5: 'Sandal',
                   6: 'Shirt',
                   7: 'Sneaker',
                   8: 'Bag',
                   9: 'Ankle boot'
                   }
    for label in np.unique(labels):
        indices = labels == label
        plt.scatter(lat_tsne[indices, 0], lat_tsne[indices, 1], label=f'Class {class_names[label]}', alpha=0.6)


    plt.title(plot_title)
    plt.legend()
    plt.savefig(file_name)
    plt.clf()

def create_dataloaders(data_path: str = 'dataset', **dl_args):
    """
    :param data_path: path to the location of the dataset
    :param dl_args: arguments that will be passed to the dataloader (for example: batch_size=32 to change the batch size)
    :return: DataLoaders for the train and test sets
    """
    # train_ds = SignLanguageDataset(os.path.join(data_path, 'sign_mnist_train.csv'))
    # test_ds = SignLanguageDataset(os.path.join(data_path, 'sign_mnist_test.csv'))
    train_ds = FashionDataset(os.path.join(data_path, 'fashion-mnist_train.csv'))
    test_ds = FashionDataset(os.path.join(data_path, 'fashion-mnist_test.csv'))
    train_dl = DataLoader(train_ds, **dl_args)
    test_dl = DataLoader(test_ds, **dl_args)
    return train_ds, train_dl, test_ds, test_dl


class FashionDataset(Dataset):
    def __init__(self, data_path):
        data = torch.from_numpy(pandas.read_csv(data_path).values)[:1000]
        # self.X = data[:, 1:].reshape(-1, 28, 28)
        self.X = data[:, 1:].reshape(-1, 28, 28).float() / 255.0  # Normalize to [0, 1]

        self.y = data[:, 0]

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, item):
        """
        :param item: index of requested item
        :return: the index and the item
        """
        return item, self.X[item]
