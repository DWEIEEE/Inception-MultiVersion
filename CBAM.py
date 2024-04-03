import torch
from torchvision import transforms
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn
import numpy as np
import scipy.io
import os

class HyperLoader(Dataset):
    def __init__(self,path,transforms):
        super(HyperLoader,self).__init__()
        self.path = path
        self.transforms = transforms
        self.folder = os.listdir(path)
        self.num_class = len(self.folder)
        self.total_pcs = self.num_count()
        self.label_mapping = {label : idx for idx, label in enumerate(self.folder)}

    def num_count(self):
        num = 0
        self.filelist = []
        for i in range(self.num_class):
            tmp = os.listdir(os.path.join(self.path,self.folder[i]))
            for tmp_name in tmp:
                self.filelist.append(os.path.join(self.path,self.folder[i],tmp_name))
            num += len(tmp)
        return num

    def __len__(self):
        return self.total_pcs

    def __getitem__(self,index):
        file = self.filelist[index]
        tmp = scipy.io.loadmat(file)
        img = tmp['cube']
        img = self.transforms(img).to(torch.float32)
        directory, file_name = os.path.split(file)
        parent_directory, sub_directory = os.path.split(directory)
        label = sub_directory
        return img, self.label_mapping[label]

def main(kwargs):
    torch.cuda.init()
    torch.manual_seed(kwargs['seed'])
    torch.cuda.manual_seed_all(kwargs['seed'])
    device = torch.device("cuda")
    cudnn.benchmark = True
    data_transform = transforms.Compose([transforms.ToTensor()])
    dataset = HyperLoader(kwargs['dataset_path'],data_transform)

if __name__ == "__main__":
    kwargs = {'dataset_path' : "E:/AutoCrop/Crop_dataset/mailiao_20231228_Cropdata_1", 'seed' : 1}
    main(kwargs)