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

    net = Model(opt)
    net = net.to(opt.device)

    train_dataset = PlayDataset(opt,is_train=True, train_val=0.9, transform=transform_test,max_label_length=opt.max_label_length)
    val_dataset = PlayDataset(opt,is_train=False, train_val=0.9, transform=transform_test,max_label_length=opt.max_label_length)

    train_dataloader=DataLoader(train_dataset,batch_size=32,shuffle=True,num_workers=8)
    val_dataloader=DataLoader(val_dataset,batch_size=32,shuffle=True,num_workers=8)

    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(opt.device)
        converter=CTCLabelConverter(opt)
    else:
        criterion=torch.nn.CrossEntroptyLoss(ignore_index=0).to(opt.device)
        converter=AttnLabelConverter(opt)
    loss_avg = Averager()
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)

    start_iter=0
    start_time=time.time()

    min_val_loss=100000000
    best_val_acc=-1

    '''
    设置loss曲线
    '''
    train_curve=list()
    valid_curve=list()

    '''加载部分权重模型'''
    # if opt.checkpoint:
    #     checkpoint=torch.load(opt.checkpoint)
    #     net.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     start_iter=checkpoint['iter']
    #     min_val_loss=checkpoint['val_loss']
    #     best_val_acc=checkpoint['val_acc']

    i=start_iter
    '''train part'''
    net.train()
    while(True):
        #train part
        batch_iter=iter(train_dataloader)
        batch=next(batch_iter)

        images = batch['image'].to(opt.device)  # images [b ,c,h,w]
        labels = batch['label']  # labels 1xb 字符数组
        text, length = converter.encode(labels, batch_max_length=opt.max_label_length)
        batch_size = images.size(0)

        if 'CTC' in opt.Prediction:
            preds = net(images, text).log_softmax(2)  #
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)  # 1xbatch_size
            preds = preds.permute(1, 0, 2)
            cost = criterion(preds, text.to(opt.device), preds_size.to(opt.device), length.to(opt.device))
        else:
            preds = net(images, text[:, :-1])
            target = text[:, 1:]
            cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

        net.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm(net.parameters(), opt.grad_clip)
        optimizer.step()
        loss_avg.add(cost)

    if i % opt.valInteral==0:
        elapsed_time=time.time()-start_time
        net.eval()
        with torch.no_grad():
            valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels, infer_time, length_of_data = validation(
                net, criterion, val_dataloader, converter, opt)
        net.train()


        # log
        loss_log = f'[{i}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
        loss_avg.reset()

        current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.2f}'
        print(current_model_log)

    if (i+1)%1e+5==0:
        torch.save(
            model.state_dict(),f'./checkpoint/{opt.exp_name}/iter_{i+1}.pth'
        )
    if i==opt.num_iter:
        print('end of train')
        sys.exit()
    i+=1









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
    parser.add_argument('--exp_name',help='where to store model')
    parser.add_argument('--max_label_length',type=int,default=35,help='the max lenghth of the label')
    parser.add_argument('--num_iter',type=int,default=30000,help='number of iterations to train for')
    parser.add_argument('--grad_clip',type=int,default=5,help='grad_clip')
    parser.add_argument('--device',type=str,default='cuda:1',help='gpu device')
    parser.add_argument('--img_rgb',type=bool,default=False,help='whether is rgb')
    '''path'''
    parser.add_argument('--dict_path',type=str,default=r'/home/luoyc/zhulin/textDR/utils/dict.txt',help='dict path')
    parser.add_argument('--img_dir',type=str,default=r'/home/luoyc/zhulin/img_crop',help='the image folder dir')
    parser.add_argument('--num_txt_path',type=str,default=r'/home/luoyc/zhulin/textDR/utils/num_train.txt',help='num_txt_path')
    parser.add_argument('--text_txt_path',type=str,default=r'/home/luoyc/zhulin/textDR/utils/text_train.txt',help='text_txt_path')
    parser.add_argument('--checkpoint',type=str,default=r'home/luoyc/zhulin/textDR/checkpoint',help='the path of the saved weights')
    parser.add_argument('--valInteravl',type=int,default=300,help='the val Interavl')
    opt = parser.parse_args()
    opt.num_class = 1556
    if not opt.exp_name:
        opt.experiment_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
    os.makedirs(f'./checkpoint/{opt.exp_name}',exist_ok=True)
    train(opt)
















