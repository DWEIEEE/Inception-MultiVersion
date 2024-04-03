import os
import numpy as np
import tifffile
import shutil
import matplotlib.pyplot as plt
import scipy.io

def process(path,FolderName):
    if os.path.exists(FolderName):
        shutil.rmtree(FolderName)
    os.makedirs(FolderName)
    if os.path.exists(path):
        items = os.listdir(path)
        for item in items:
            os.makedirs(os.path.join(FolderName,item))
            tmp = os.listdir(os.path.join(path,item))
            for i in range(0,len(tmp)):
                convertTIF(os.path.join(path,item,tmp[i]),os.path.join(FolderName,item))
        return 0
    else:
        print("Dataset Path Error")
        return 0

def convertTIF(path,outpath):
    folder, filename = os.path.split(path)
    file_name = os.path.splitext(os.path.basename(filename))[0]
    tmp = scipy.io.loadmat(path)
    img = tmp["cube"]
    tif_file_path = os.path.join(outpath, file_name + '.tif')
    tifffile.imwrite(tif_file_path, img)

def showTIF(path):
    tif_data = tifffile.imread(path)
    print(tif_data.shape)
    plt.imshow(tif_data[:, :, 23], cmap='gray')
    plt.title('TIF Image')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    #2dataset_path = "20231108_data"
    #FolderName = "20231108_TIFdata"
    #process(dataset_path,FolderName)
    filepath = "20231108_TIFdata/normal/HPCCD_00012_2023_1108_095332_RT_New1.tif"
    showTIF(filepath)