import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from datetime import datetime
import torch
import seaborn
from sklearn.metrics import confusion_matrix, f1_score
import os
import pickle

def showInfo(path,figure = False):
    total_pcs = 0
    if os.path.exists(path):
        items = os.listdir(path)
        for index, item in enumerate(items):
            tmp = os.listdir(os.path.join(path,item))
            total_pcs += len(tmp)
            if figure == True:
                for i in range(0,len(tmp)):
                    showData(os.path.join(path,item,tmp[i]))
            print(f"[{index + 1}] {item} : {len(tmp)}")
        print(f"=> {len(items)} class : {total_pcs} pcs")
        return 0
    else:
        print("Dataset Path Error")
        return 0

def showData(path,band = 23):
    folder, filename = os.path.split(path)
    tmp = scipy.io.loadmat(path)
    img = tmp["cube"][:,:,band]
    print(f"filename : {filename}, size : {img.shape}, max : {np.nanmax(img)}, min : {np.nanmin(img)}")
    plt.imshow(img,cmap='gray')
    plt.title(filename)
    plt.show()

def getTimestamp():
    current_datetime = datetime.now()
    date = current_datetime.strftime("%Y%m%d")
    time = current_datetime.strftime("%H%M")
    timestamp = f"{date}{time}"
    return current_datetime, timestamp

def checkFolder(path):
    parts = path.split('/')
    os.makedirs(parts[0]+"/"+parts[1], exist_ok=True)
    os.makedirs(path, exist_ok=True)

def getConfusionmatrix(model,device,dataset,classlist,outPath,labelMap = []):
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for (images, targets) in dataset:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            _, predicted_classes = torch.max(outputs, dim=1)
            all_predictions.extend(predicted_classes.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    if len(labelMap) != 0:
        reversed_classlist = {v: k for k, v in classlist.items()}
        print(f"reverse : {reversed_classlist}")
        tmp = []
        for value in all_predictions:
            try:
                print(f"1 : {value}")
                print(f"2 : {reversed_classlist[value]}")
                print(f"3 : {labelMap[reversed_classlist[value]]}")
                tmp.append(labelMap[reversed_classlist[value]])
            except:
                tmp.append(len(labelMap))
        all_predictions = tmp
        labelMap['Others'] = len(labelMap)
    #print(f"predictions : {all_predictions}")
    #print(f"labels : {all_labels}")
    correct = (all_predictions == all_labels).sum().item()
    accuracy = correct / len(dataset.dataset)
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    f1score = f1_score(all_labels, all_predictions, average='weighted')

    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f'accuracy : {accuracy}')
    print(f'F1-score: {f1score}')

    plt.figure(figsize=(8, 6))
    if len(labelMap) != 0:
        seaborn.heatmap(conf_matrix, annot=True, cbar=False, fmt="d", cmap="Blues", xticklabels=labelMap, yticklabels=labelMap)
    else:
        seaborn.heatmap(conf_matrix, annot=True, cbar=False, fmt="d", cmap="Blues", xticklabels=classlist, yticklabels=classlist) #[f"Class {i}" for i in range(conf_matrix.shape[0])]
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.title(f"Confusion Matrix | accuracy : {round(accuracy*100,2)}% | F1 : {round(f1score*100,2)}")
    plt.savefig(outPath + '/Confusion_Matrix.png')
    plt.show()

def getExecutime(execution_time):
    total_seconds = execution_time.total_seconds()
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return int(hours), int(minutes), int(seconds)

def writeDetail(kwargs,best_list,record,execution_time,path,label_mapping):
    hours, minutes, seconds = getExecutime(execution_time)
    if os.path.exists(path+'/detail.txt'):
        os.remove(path+'/detail.txt')
    with open(path+"/class.json", "wb") as fp:
        pickle.dump(label_mapping, fp)
    list_item = ['Train Loss : ','Valid Loss : ','Test Loss : ','Train Accuracy : ','Valid Accuracy : ','Test Accuracy : ']
    with open(path+'/detail.txt', "w") as file:
        for key, value in kwargs.items():
            file.write('%s: %s\n' % (key, value))
        file.write(f"\n[Best at {record} epoch]\n")
        for i in range(len(list_item)):
            file.write(f'{list_item[i]}{best_list[i]}\n')
        file.write(f"\nExecution time : {hours} hr {minutes} min {seconds} sec.")

def drawFig(train_loss, train_accuracy, valid_loss, valid_accuracy, epochs, record, outPath, save = False):
    epoch_list = list(range(epochs))
    plt.figure(1)
    plt.plot(epoch_list, train_accuracy, label='train_accuracy')
    plt.plot(epoch_list, valid_accuracy, label='valid_accuracy')
    plt.scatter(record, valid_accuracy[record], color='black', marker='o', zorder=10)
    plt.text(record, valid_accuracy[record]+0.05, f'Final', fontsize=10, va='center', ha='center', color='black')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 100.)
    if save == True:
        plt.savefig(outPath + '/Training_Accuracy.png')

    plt.figure(2)
    plt.plot(epoch_list, train_loss, label='train_loss')
    plt.plot(epoch_list, valid_loss, label='valid_loss')
    plt.scatter(record, valid_loss[record], color='black', marker='o', zorder=10)
    plt.text(record, valid_loss[record]+0.05, f'Final', fontsize=10, va='center', ha='center', color='black')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    if save == True:
        plt.savefig(outPath + '/Training_Loss.png')

    plt.show()

def discover(dataset_path):
    dataset_filename = os.listdir(dataset_path)
    class_dict = {}
    time_dict = {}
    total_number = 0
    print("\n================================================================")
    for dataset_name in dataset_filename:
        folder_name = os.listdir(os.path.join(dataset_path,dataset_name))
        for class_name in folder_name:
            try:
                class_dict[class_name] = class_dict[class_name] + len(os.listdir(os.path.join(dataset_path,dataset_name,class_name)))
                time_dict[class_name] = time_dict[class_name] + 1
            except:
                class_dict[class_name] = len(os.listdir(os.path.join(dataset_path,dataset_name,class_name)))
                time_dict[class_name] = 1
            total_number = total_number + len(os.listdir(os.path.join(dataset_path,dataset_name,class_name)))
    for idx, (ClassName, number) in enumerate(class_dict.items()):
        print(f"[{(idx+1):2d}] ({time_dict[ClassName]}) {ClassName:30s} {number}")
    print("================================================================")
    print(f"Total : {total_number}\n")

def GPUinfo():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"可用的 GPU 數量為: {gpu_count}")

        for i in range(gpu_count):
            gpu = torch.cuda.get_device_properties(i)
            print(f"GPU {i} 的資訊:")
            print(f"  名稱: {gpu.name}")
            print(f"  記憶體容量: {gpu.total_memory / 1024**3:.1f} GB")
    else:
        print("未偵測到可用的 GPU")
