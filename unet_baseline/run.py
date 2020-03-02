import os


os.system("/home/xpeng/anaconda3/envs/py36/bin/python train.py --tensorboard_name rectanglemask_size_loss_1 --params_name size_loss1.pkl --gpu_order=0")
os.system("/home/xpeng/anaconda3/envs/py36/bin/python train.py --tensorboard_name rectanglemask_size_loss_2 --params_name size_loss2.pkl --gpu_order=0")
os.system("/home/xpeng/anaconda3/envs/py36/bin/python train.py --tensorboard_name rectanglemask_size_loss_3 --params_name size_loss3.pkl --gpu_order=0")
os.system("/home/xpeng/anaconda3/envs/py36/bin/python train.py --tensorboard_name rectanglemask_size_loss_4 --params_name size_loss4.pkl --gpu_order=0")
os.system("/home/xpeng/anaconda3/envs/py36/bin/python train.py --tensorboard_name rectanglemask_size_loss_5 --params_name size_loss5.pkl --gpu_order=0")


# import requests
#
# url = "https://sc.ftqq.com/SCU28703Te109f3ff3fede315f4017d79786274ab5b35cf275612b.send?"
#
# params = {"text":"polyp_seg result",
#           'desp':''}
#
# res = requests.get(url=url,params=params)
#
# print(res.text)

#/home/xpeng/anaconda3/envs/py36/python


