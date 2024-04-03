import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from func import getConfusionmatrix
import torch.backends.cudnn as cudnn
import pickle
import os
from func import getTimestamp, checkFolder
from Model import Inception
from process import HyperLoader

class main:
    def __init__(self,kwargs):
        self.model_path = kwargs['model']
        self.dataset = kwargs['dataset']
        self.outpath = kwargs['outPath']
        self.savefig = kwargs['saveFig']
        json = kwargs['JSON']
        self.predict_class = kwargs['predict_class']
        self.device = torch.device("cuda")
        with open(json, "rb") as fp:
            self.classlist = pickle.load(fp)

    def execute(self):
        torch.cuda.init()
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        cudnn.benchmark = True

        start_time, timestamp = getTimestamp()
        print(f"TimeStamp : {timestamp}")

        model_parts = self.model_path.split('/')
        dataset_parts = self.dataset.split('/')
        self.outpath = os.path.join(self.outpath + "/" + f"{model_parts[1]}_to_{dataset_parts[-1]}" + "/" + timestamp)
        checkFolder(self.outpath)

        model = Inception(len(self.classlist)).to(self.device)
        model_weights = torch.load(self.model_path)
        model.load_state_dict(model_weights)

        data_transform = transforms.Compose([transforms.ToTensor()])
        dataset = HyperLoader(self.dataset,data_transform,False,self.predict_class)
        TestData = DataLoader(dataset, 1)
        print(f"dataset samples: {len(dataset)}")
        print(f"label_map: {dataset.label_mapping}")
        print(f"classlist: {self.classlist}")
        getConfusionmatrix(model,self.device,TestData,self.classlist,self.outpath,dataset.label_mapping)

if __name__ == "__main__":
    kwargs = {
              "model":"result/TaiBao_3/202403171632/model.pt",
              "dataset":"E:/AutoCrop/Crop_dataset/MaiLiao_2",
              "predict_class":["Health","Cold Injury","Aspergillus Fumigatus"],
              "JSON":"result/TaiBao_3/202403171632/class.json",
              "outPath":"test",
              "saveFig":True
            }
    work = main(kwargs)
    work.execute()
