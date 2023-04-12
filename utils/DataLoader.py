from torch.utils.data import DataLoader
import torchvision

def load_data_cifar10(path_to_data: str, download=True, batch_size=128, shuffle=True, num_workers=1):
    # data pre-processing
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(size=32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test =  transform_train = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # load datasets
    trainset = torchvision.datasets.CIFAR10(
        root=path_to_data, train=True, download=download, transform=transform_train
    )
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    
    testset = torchvision.datasets.CIFAR10(
        root=path_to_data, train=False, download=download, transform=transform_test
    )
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader, classes

