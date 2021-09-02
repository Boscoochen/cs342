from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import csv
import os

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
        """
        Your code here
        Hint: Use the python csv library to parse labels.csv
        
        WARNING: Do not perform data normalization here. 
        """
        self.dataset_path = dataset_path
        
        #dataset_path = "data/train/" 
        self.img_path = []
        self.main_path = []
        self.img_label = []
        
        label_csv_path = os.path.join(dataset_path, "labels.csv")
        with open(label_csv_path) as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)
            for line in csv_reader:
                self.img_path.append(line[0])
                
                self.img_label.append(LABEL_NAMES.index(line[1]))
        for path in self.img_path:
            self.main_path.append(os.path.join(self.dataset_path, path))
        #print(self.main_path)
        
        #raise NotImplementedError('SuperTuxDataset.__init__')

    def __len__(self):
        """
        Your code here
        """
        return len(self.img_path)
        raise NotImplementedError('SuperTuxDataset.__len__')

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """
        img_item_path = self.main_path[idx]
        img = Image.open(img_item_path)
        #transforms.Compose([transforms.Resize(3,64,64), transforms.ToTensor(),transforms.Normalize([0],[1])])
        trans_totensor = transforms.ToTensor()
        img_tensor = trans_totensor(img)
        img_tensor = img_tensor.reshape(3, 64, 64)
        
        label = self.img_label[idx]
        return img_tensor, label
        raise NotImplementedError('SuperTuxDataset.__getitem__')


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
