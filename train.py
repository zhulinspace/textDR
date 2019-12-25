import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import time
import argparse
from test import validation
from dataset import PlayDataset
from model import Model
from utils.convert_lossavg import CTCLabelConverter,AttnLabelConverter,Averager

import warnings
warnings.filterwarnings("ignore")

# device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

transform_test = transforms.Compose([
    transforms.Resize((32,640)),
    transforms.ToTensor()
])

def train(opt):
    device=opt.device
    net = Model(opt)
    net = net.to(device)

    train_dataset = PlayDataset(opt,is_train=True, train_val=0.8, transform=transform_test,max_label_length=opt.max_label_length)
    val_dataset = PlayDataset(opt,is_train=False, train_val=0.8, transform=transform_test,max_label_length=opt.max_label_length)
    datasets={'train':train_dataset,'val':val_dataset}

    dataloaders={phase:DataLoader(dataset=datasets[phase],batch_size=8,shuffle=True,num_workers=8) for phase in ['train','val']}

    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
        converter=CTCLabelConverter(opt)
    else:
        criterion=torch.nn.CrossEntroptyLoss(ignore_index=0).to(device)
        converter=AttnLabelConverter(opt)
    loss_avg = Averager()
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))

    start_epoch=0
    min_val_loss=100000000
    best_val_acc=-1

    '''加载部分权重模型'''
    if opt.checkpoint:
        checkpoint=torch.load(opt.checkpoint)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch=checkpoint['epoch']
        min_val_loss=checkpoint['val_loss']
        best_val_acc=checkpoint['val_acc']


    '''多gpu部分'''
    # save model
    state = {
            'net': net.state_dict(),
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/first.t7')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, default="None", help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default="ResNet", help='FeatureExtraction stage. VGG|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default="BiLSTM", help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default="CTC", help='Prediction stage. CTC|Attn')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    parser.add_argument('--max_label_length',type=int,default=35,help='the max lenghth of the label')
    parser.add_argument('--epoch',type=int,default=100,help='number of iterations to train for')
    parser.add_argument('--grad_clip',type=int,default=5,help='grad_clip')
    parser.add_argument('--device',type=str,default='cuda:1',help='gpu device')
    parser.add_argument('--img_rgb',type=bool,default=False,help='whether is rgb')
    '''path'''
    parser.add_argument('--dict_path',type=str,default=r'/home/luoyc/zhulin/textDR/utils/dict.txt',help='dict path')
    parser.add_argument('--img_dir',type=str,default=r'/home/luoyc/zhulin/img_crop',help='the image folder dir')
    parser.add_argument('--num_txt_path',type=str,default=r'/home/luoyc/zhulin/textDR/utils/num_train.txt',help='num_txt_path')
    parser.add_argument('text_txt_path',type=str,default=r'/home/luoyc/zhulin/textDR/utils/text_train.txt',help='text_txt_path')
    parser.add_argument('checkpoint',type=str,default=r'home/luoyc/zhulin/textDR/checkpoint/weights/',help='the path of the saved weights')

    opt = parser.parse_args()
    opt.num_class = 1556
    train(opt)
















