import os
import torch
import pandas as pd
import random

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from utils.make_txt_dict import get_dict
import argparse


class PlayDataset(Dataset):

    def __init__(self, opt,is_train=True, train_val=0.9, transform=None,max_label_length=35):
        #max_label_length为最大的标签长度 即getitem()中的label是要求shape一样

        '''
               若是text_txt_path img_label_list=[('图片路径'，['文字标签']),(),...]
               若是num_txt_path image_label_list=[('图片路径'，[1,2,3,....]),(),...]
        '''

        self.max_label_length=max_label_length
        self.img_label_list=self.read_text_file(opt.text_txt_path)
        # self.img_label_list=self.read_num_file(opt.num_txt_path)
        self.img_rgb=opt.img_rgb


        self.is_train = is_train
        self.train_val = train_val
        self.transform = transform
        # self.char_to_id,self.id_to_char=get_dict(opt.dict_path)

    def read_text_file(self,label_file):
        img_label_list = []
        with open(label_file, 'r')as f:
            lines = f.readlines()
            for line in lines:
                content = line.rstrip().split(' ')
                imgdir = content[0]
                labels = content[1]
                # for value in content[1:]:
                #     labels.append(str(value))
                img_label_list.append((imgdir, labels))
        # print('img_lable_list',img_label_list)
        return img_label_list

    def read_num_file(self,label_file):
        '''
        :param label_file: 标签文件
        :return: 元组序列 （图片完整路径，标签）
        '''
        img_label_list=[]
        with open(label_file,'r')as f:
            lines=f.readlines()
            for line in lines:
                content=line.rstrip().split(' ')
                imgdir=content[0]
                labels=[]
                for value in content[1:]:
                    labels.append(int(value))
                img_label_list.append((imgdir,labels))
        # print('img_lable_list',img_label_list)
        return img_label_list


    def __len__(self):
        if self.is_train:
            return int(len(self.img_label_list) * self.train_val)
        else:
            return len(self.img_label_list) - int(len(self.img_label_list) * self.train_val)



    def __getitem__(self, idx):
        '''
        :param idx:
        :return: {图片，标签，真实标签长度}
        '''
        if not self.is_train:
            idx += int(len(self.img_label_list) * self.train_val)

        img_path,real_label = self.img_label_list[idx]
        # print('imgpath',img_path)
        # print('real_label',real_label)

        if self.img_rgb:
            image=Image.open(img_path)
        else:
            image=Image.open(img_path).convert('L')


        if self.transform:
            image = self.transform(image)

        '''num_txt:'''
        # label = np.zeros((self.max_label_length))
        # for idx,num in enumerate(real_label):
        #     label[idx]=num

        '''text_txt'''
        label=real_label

        sample = {
            'image': image,
            'label': label,
            # 'real_length':len(real_label)
        }
        return sample

    
if __name__ == "__main__":
    #test
    '''
    目标是生成text_label 
    即从一个batch里取出来的大小是 [batch_size]
    text=['rts','hu','test','huhu','hu']
    '''
    transform_test = transforms.Compose([
        transforms.Resize((32,640)),# h ,w
        transforms.ToTensor(),
    ])#scale是在PIL图片上基础上做的，即Totensor放在最后，另外Norm放在totensor之后
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_txt_path',type=str,default=r'/home/luoyc/zhulin/textDR/utils/text_train.txt',help='text_txt_path')
    parser.add_argument('--img_rgb',type=bool,default=False,help='whether is rgb')
    opt = parser.parse_args()



    dataset = PlayDataset(opt,is_train=True, train_val=1, transform=transform_test)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=8)
    for i, sample_batch in enumerate(dataloader):
        print(i, sample_batch['image'].shape)
        print(i, sample_batch['label'])
        # print(i,sample_batch['real_length'].shape)

