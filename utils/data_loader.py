import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from utils.My_Enums import DatasetPicker as Dp


class DatasetLoader(object):
    @staticmethod
    def get_data_loader(dataset: Dp, dataroot: str, batch_size: int, convolutional: bool, download=True):
        if convolutional:
           trans1 = transforms.Compose([
                transforms.Scale(32),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
            ])
           trans2 = transforms.Compose([
                transforms.Scale(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            trans1 = transforms.Compose([
                transforms.Resize(32),
                transforms.Resize((1, 1024)),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5)
            ])
            trans2 = transforms.Compose([
                transforms.Resize(32),
                transforms.Resize((1, 1024)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        if dataset == Dp.MNIST:
            # trans = transforms.Compose([
            #     transforms.Resize(32),
            #     transforms.Resize((1, 1024)),
            #     transforms.ToTensor(),
            #     transforms.Normalize(0.5, 0.5),
            # ])
            train_dataset = dset.MNIST(root=dataroot, train=True, download=download, transform=trans1)
            test_dataset = dset.MNIST(root=dataroot, train=False, download=download, transform=trans1)

        elif dataset == Dp.FASHION_MNIST:
            trans = transforms.Compose([
                transforms.Resize(32),
                transforms.Resize((1, 1024)),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
            ])
            train_dataset = dset.FashionMNIST(root=dataroot, train=True, download=download, transform=trans1)
            test_dataset = dset.FashionMNIST(root=dataroot, train=False, download=download, transform=trans1)

        elif dataset == Dp.CIFAR10:
            trans = transforms.Compose([
                transforms.Resize(32),
                transforms.Resize((1, 1024)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            train_dataset = dset.CIFAR10(root=dataroot, train=True, download=download, transform=trans2)
            test_dataset = dset.CIFAR10(root=dataroot, train=False, download=download, transform=trans2)

        elif dataset == Dp.STL10:
            trans = transforms.Compose([
                transforms.Resize(32),
                transforms.Resize((1, 1024)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            train_dataset = dset.STL10(root=dataroot, train=True, download=download, transform=trans2)
            test_dataset = dset.STL10(root=dataroot, train=False, download=download, transform=trans2)

        # Check if everything is ok with loading datasets
        assert train_dataset
        assert test_dataset

        train_dataloader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = data_utils.DataLoader(test_dataset,  batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader
