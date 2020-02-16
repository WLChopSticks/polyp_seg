import os


def rename_func(path):
    for file in os.listdir(path):
        if '300' in path:
            fileNew = 'cvc-300_'+file
        elif '612' in path:
            fileNew =  'cvc-612_'+file
        print("Old:", file, "New", fileNew)
        if(fileNew != file):
            os.rename(os.path.join(path,file), os.path.join(path,fileNew))

path2 = r"C:\Users\wang\Desktop\框数据集\框数据集\实验3产生矩形mask已确认\CVC-300"
path3 = r"C:\Users\wang\Desktop\框数据集\框数据集\实验3产生矩形mask已确认\CVC-612"


rename_func(path2)
rename_func(path3)





