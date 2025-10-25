import torch
import torchvision
import torchvision.transforms as transforms
from collections import defaultdict

#from torchvision.transforms import v2
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

def get_dataloaders(test_bs=1):

    torch.manual_seed(42)

    train_transform, test_transform = get_transform()

    train_ds = torchvision.datasets.CIFAR10('/leonardo_work/EUHPC_A04_051/datasets', train=True, transform=train_transform, download=True)
    test_ds = torchvision.datasets.CIFAR10('/leonardo_work/EUHPC_A04_051/datasets', train=False, transform=test_transform, download=True)
    num_classes = 10

    ###
    '''
    k = 100
    train_indices = get_subset_indices_per_class(train_ds, k, num_classes)
    test_indices = get_subset_indices_per_class(test_ds, k, num_classes)

    train_ds = torch.utils.data.Subset(train_ds, train_indices)
    test_ds = torch.utils.data.Subset(test_ds, test_indices)
    '''
    ###

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Total number of samples in train_dl: {len(train_dl.dataset)}")

    return train_dl, test_dl

def get_transform():
    padding=4
    size = 32
    mean, std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]

    train_transform_list = [transforms.RandomCrop(size=(size,size), padding=padding)]
    train_transform_list.append(transforms.RandomCrop(size=(size,size), padding=padding))

    #train_transform_list.append(v2.AutoAugmentPolicy.CIFAR10)
    train_transform_list.append(AutoAugment(policy=AutoAugmentPolicy.CIFAR10))

    train_transform = transforms.Compose(
        train_transform_list+[
            transforms.ToTensor(),
            #transforms.Resize(16, antialias=True),
            transforms.Normalize(
                mean= mean,
                std = std
            )
        ]
    )
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Resize(16, antialias=True),
        transforms.Normalize(
            mean= mean,
            std = std
        )
    ])

    return train_transform, test_transform

def get_subset_indices_per_class(dataset, k, num_classes=10):
    """Get indices of k samples per class from the dataset."""
    class_indices = defaultdict(list)

    for idx, (_, label) in enumerate(dataset):
        if len(class_indices[label]) < k:
            class_indices[label].append(idx)
        if all(len(class_indices[c]) == k for c in range(num_classes)):
            break

    subset_indices = [idx for indices in class_indices.values() for idx in indices]
    return subset_indices

class FilteredCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, classes_to_keep=None):
        super(FilteredCIFAR10, self).__init__(root, train=train, transform=transform,
                                              target_transform=target_transform, download=download)
        if classes_to_keep is not None:
            self.classes_to_keep = classes_to_keep
            self.class_to_idx = {cls: idx for idx, cls in enumerate(classes_to_keep)}
            self._filter_classes()

    def _filter_classes(self):
        targets = torch.tensor(self.targets)
        mask = torch.isin(targets, torch.tensor([self.class_to_idx[cls] for cls in self.classes_to_keep]))
        self.data = self.data[mask.numpy()]
        self.targets = targets[mask].tolist()
        target_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(set(self.targets)))}
        self.targets = [target_map[target] for target in self.targets]

def get_filtered_dataloaders(NUM_CLASSES):
    train_transform, test_transform = get_transform()

    classes_to_keep = ['plane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'][:NUM_CLASSES]
    
    train_ds = FilteredCIFAR10('/leonardo_work/EUHPC_A04_051/datasets', train=True, transform=train_transform, download=True, classes_to_keep=classes_to_keep)
    test_ds = FilteredCIFAR10('/leonardo_work/EUHPC_A04_051/datasets', train=False, transform=test_transform, download=True, classes_to_keep=classes_to_keep)

    num_classes = len(classes_to_keep)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    return train_dl, test_dl


def get_transform():
    padding=4
    size = 32
    mean, std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]

    train_transform_list = [transforms.RandomCrop(size=(size,size), padding=padding)]
    train_transform_list.append(transforms.RandomCrop(size=(size,size), padding=padding))

    #train_transform_list.append(v2.AutoAugmentPolicy.CIFAR10)
    train_transform_list.append(AutoAugment(policy=AutoAugmentPolicy.CIFAR10))

    train_transform = transforms.Compose(
        train_transform_list+[
            transforms.ToTensor(),
            #transforms.Resize(64, antialias=True),
            transforms.Normalize(
                mean= mean,
                std = std
            )
        ]
    )
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Resize(64, antialias=True),
        transforms.Normalize(
            mean= mean,
            std = std
        )
    ])

    return train_transform, test_transform
