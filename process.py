import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import scipy.io
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
from Model import ourModel, Inception, CBAM_Inception, Skip_Inception
import torch.optim as optim
import torch.nn as nn
from func import drawFig, getTimestamp, getConfusionmatrix, checkFolder, writeDetail
from torch.optim.lr_scheduler import StepLR
from func import showInfo
import numpy as np
import cv2 as cv
import torch.backends.cudnn as cudnn

class HyperLoader(Dataset):
    def __init__(self,path,transforms,rotation=False,restriction=[]):
        super(HyperLoader,self).__init__()
        self.path = path
        self.rotation = rotation
        self.transforms = transforms
        self.restriction = restriction
        self.folder = os.listdir(path)
        self.num_class = len(self.folder)
        if len(self.restriction) != 0:
            self.num_class = len(self.restriction)
            self.folder = self.restriction
        self.total_pcs = self.num_count()
        self.label_mapping = {label : idx for idx, label in enumerate(self.folder)}

    def rotate(self,img):
        angle = np.random.choice([0, 90, 180, 270])
        rotated_img = np.zeros_like(img)
        for i in range(img.shape[2]):
            if angle == 0:
                rotated_img[:,:,i] = img[:,:,i]
            else:
                rotated_img[:,:,i] = cv.rotate(img[:,:,i], cv.ROTATE_90_CLOCKWISE if angle == 90 else
                cv.ROTATE_180 if angle == 180 else
                cv.ROTATE_90_COUNTERCLOCKWISE if angle == 270 else 0)
        return rotated_img

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
        if self.rotation == True:
            img = self.rotate(img)
        img = self.transforms(img).to(torch.float32)
        directory, file_name = os.path.split(file)
        parent_directory, sub_directory = os.path.split(directory)
        label = sub_directory
        return img, self.label_mapping[label]

def count(dataset, title):
    class_counts = Counter()
    for _, targets in dataset:
        class_counts.update(targets)
    print(title)
    for index, (class_label, count) in enumerate(class_counts.items()):
        print(f"Class {index+1} {class_label} : {count} samples")
    return 0

def train(model, device, train_dataset, optimizer, epoch, log_interval):
    model.train()
    running_loss = 0
    correct = 0

    criterion = nn.CrossEntropyLoss()

    for batch_idx, (images, targets) in enumerate(train_dataset):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted_classes = torch.max(outputs, dim=1)
        correct += (predicted_classes == targets).sum().item()
        if batch_idx % log_interval == 0:
            print(f'[{epoch + 1}, {batch_idx + 1:3d}] loss: {loss.item()}')
    accuracy = correct / len(train_dataset.dataset)
    total_loss = running_loss/len(train_dataset)
    print(f'[{epoch + 1}] total Average loss: {total_loss}   Accuracy : {round(accuracy*100,2)}%')
    return total_loss, round(accuracy*100,2)

def valid(model, device, valid_dataset):
    model.eval()
    running_loss = 0
    correct = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for (images, targets) in valid_dataset:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            running_loss += criterion(outputs, targets)
            _, predicted_classes = torch.max(outputs, dim=1)
            correct += (predicted_classes == targets).sum().item()

    accuracy = correct / len(valid_dataset.dataset)
    total_loss = running_loss / len(valid_dataset)
    print(f'Valid Average loss: {total_loss}  Accuracy : {round(accuracy*100,2)}%')
    return total_loss, round(accuracy*100,2)

def test(model, device, test_dataset):
    model.eval()
    running_loss = 0
    correct = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for (images, targets) in test_dataset:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            running_loss += criterion(outputs, targets)
            _, predicted_classes = torch.max(outputs, dim=1)
            correct += (predicted_classes == targets).sum().item()
            #print(f"predicted_classes : {predicted_classes}")
            #print(f"targets : {targets}")

    accuracy = correct / len(test_dataset.dataset)
    total_loss = running_loss / len(test_dataset)
    print(f'Test Average loss: {total_loss}  Accuracy : {round(accuracy*100,2)}%')
    return total_loss, round(accuracy*100,2)
def preprocess(path,kwargs):
    torch.cuda.init()
    cudnn.benchmark = True
    torch.manual_seed(kwargs['seed'])
    torch.cuda.manual_seed_all(kwargs['seed'])

    if kwargs['try'] != True:
        start_time, timestamp = getTimestamp()
    else:
        start_time, timestamp = getTimestamp()
        timestamp = "000000000000"

    print(f"TimeStamp : {timestamp}")
    kwargs['outPath'] = kwargs['outPath'] + timestamp

    checkFolder(kwargs['outPath'])

    if torch.cuda.is_available():
        print("GPU is available.")
        device = torch.device("cuda")
    else:
        print("GPU is not available.")
        device = torch.device("cpu")

    if kwargs['Normalization'] == True:
        data_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(np.zeros(24), np.ones(24))])
    else:
        data_transform = transforms.Compose([transforms.ToTensor()])

    dataset = HyperLoader(path,data_transform,kwargs['rotation'])

    features = [dataset[i][0] for i in range(len(dataset))]
    labels = [dataset[i][1] for i in range(len(dataset))]

    class_num = len(set(labels))

    sss = StratifiedShuffleSplit(n_splits=1, train_size=kwargs["train_ratio"], test_size=kwargs["valid_ratio"]+kwargs["test_ratio"], random_state=kwargs['seed'])
    train_index, rest_index = next(sss.split(features, labels))
    valid_test_ratio = kwargs["valid_ratio"] / (kwargs["valid_ratio"]+kwargs["test_ratio"])
    valid_size = int(valid_test_ratio * len(rest_index))
    valid_index, test_index = rest_index[:valid_size], rest_index[valid_size:]

    train_set = Subset(dataset, train_index)
    valid_set = Subset(dataset, valid_index)
    test_set = Subset(dataset, test_index)

    train_dataset = DataLoader(train_set, kwargs["batch_size"], kwargs["shuffle"])
    valid_dataset = DataLoader(valid_set, kwargs["batch_size"], kwargs["shuffle"])
    test_dataset = DataLoader(test_set, kwargs["batch_size"], kwargs["shuffle"])

    print(f"train samples: {len(train_set)}")
    print(f"valid samples: {len(valid_set)}")
    print(f"test samples: {len(test_set)}")
    if kwargs["model"] == "CBAM":
        model = CBAM_Inception(class_num,kwargs['dropout']).to(device)
    elif kwargs["model"] == "Skip":
        model = Skip_Inception(class_num,kwargs['dropout']).to(device)
    else:
        model = Inception(class_num,kwargs['dropout']).to(device)
    tmp_valid = 100
    record = 0
    train_lossList = []
    valid_lossList = []
    train_accuracyList = []
    valid_accuracyList = []
    optimizer = optim.Adadelta(model.parameters(), lr=kwargs['lr'], weight_decay=kwargs["weight_decay"]) # ,weight_decay=1e-3
    scheduler = StepLR(optimizer, step_size=1, gamma=kwargs['gamma'])
    for epoch in range(kwargs['epochs']):
        current_lr = optimizer.param_groups[0]['lr']
        print(f'\nEpoch {epoch+1}, Learning Rate: {current_lr}')
        train_loss, train_accuracy = train(model,device,train_dataset,optimizer,epoch,kwargs['log_interval'])
        valid_loss, valid_accuracy = valid(model,device,valid_dataset)
        test_loss, test_accuracy = test(model,device,test_dataset)
        if tmp_valid > valid_loss:
            if kwargs['saveModel'] == True:
                print("[ save new  model's weights ]")
                torch.save(model.state_dict(), kwargs['outPath'] + "/model.pt")
            best_list = [train_loss, valid_loss, test_loss, train_accuracy, valid_accuracy, test_accuracy]
            tmp_valid = valid_loss
            record = epoch

        print(f"Best at {record+1} epoch.")
        if (epoch % kwargs['gamma_gap']) == 0 and epoch != 0 and epoch != (kwargs['gamma_gap']):
            scheduler.step()

        train_lossList.append(train_loss)
        train_accuracyList.append(train_accuracy)
        valid_lossList.append(valid_loss.cpu().numpy())
        valid_accuracyList.append(valid_accuracy)

    print('Finished Training')
    drawFig(train_lossList, train_accuracyList, valid_lossList, valid_accuracyList, kwargs["epochs"], record, kwargs['outPath'], kwargs['saveFig'])

    if kwargs["model"] == "CBAM":
        test_model = CBAM_Inception(class_num,kwargs['dropout']).to(device)
    elif kwargs["model"] == "Skip":
        test_model = Skip_Inception(class_num,kwargs['dropout']).to(device)
    else:
        test_model = Inception(class_num,kwargs['dropout']).to(device)

    test_model_weights = torch.load(kwargs['outPath'] + "/model.pt")
    test_model.load_state_dict(test_model_weights)
    classlist = [key for key, value in dataset.label_mapping.items()]
    getConfusionmatrix(test_model,device,test_dataset,classlist,kwargs['outPath'])
    end_time, _ = getTimestamp()
    writeDetail(kwargs,best_list,record+1,(end_time - start_time),kwargs['outPath'],dataset.label_mapping)
    return 0

if __name__ == "__main__":
    dataset_path = "D:/Project/AutoCrop/Crop_dataset/MaiLiao_2"
    kwargs = {'batch_size': 16,
              'shuffle': True,
              'train_ratio': 0.7,
              'valid_ratio': 0.15,
              'test_ratio': 0.15,
              'epochs' : 2000,
              'log_interval' : 10,
              'seed' : 1,
              'lr': 0.002,
              'weight_decay': 0,
              "dropout": 0.4,
              'gamma': 0.5,
              'gamma_gap': 800,
              'Normalization' : False,
              'rotation': False,
              'model': "Skip",
              'outPath': 'result/MaiLiao_2/',
              'saveModel': True,
              'saveFig': True,
              "try": True,
              'additional': "Rectified"}
    #showInfo(dataset_path,False)
    preprocess(dataset_path, kwargs)