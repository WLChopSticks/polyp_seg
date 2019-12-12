import os

# prefix = 'cvc-300_'
prefix = 'cvc-612_'
data_root = '/home/jiaxin/MICCAI2020/data/Total/CVC-612/gtpolyp'

images = os.listdir(data_root)

for i in images:
    os.rename(os.path.join(data_root, i), os.path.join(data_root, prefix+i))
