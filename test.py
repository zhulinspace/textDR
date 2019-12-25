import os
import time
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from nltk.metrics.distance import edit_distance

from utils.convert_lossavg import CTCLabelConverter, AttnLabelConverter, Averager
from model import Model
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def validation(model, criterion, evaluation_loader, converter, opt):
    """ validation or evaluation """
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()
    device=opt.device

    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        # For max length prediction
        length_for_pred = torch.IntTensor([opt.max_label_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.max_label_length + 1).fill_(0).to(device)

        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.max_label_length)

        start_time = time.time()
        if 'CTC' in opt.Prediction:
            preds = model(image, text_for_pred).log_softmax(2)
            forward_time = time.time() - start_time

            # Calculate evaluation loss for CTC deocder.
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            # permute 'preds' to use CTCloss format
            cost = criterion(preds.permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)

            # Select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_index = preds_index.view(-1)
            preds_str = converter.decode(preds_index.data, preds_size.data)

        else:
            preds = model(image, text_for_pred, is_train=False)
            forward_time = time.time() - start_time

            preds = preds[:, :text_for_loss.shape[1] - 1, :]
            target = text_for_loss[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            labels = converter.decode(text_for_loss[:, 1:], length_for_loss)

        infer_time += forward_time
        valid_loss_avg.add(cost)

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            if 'Attn' in opt.Prediction:
                gt = gt[:gt.find('[s]')]
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            if pred == gt:
                n_correct += 1
            if len(gt) == 0:
                norm_ED += 1
            else:
                norm_ED += edit_distance(pred, gt) / len(gt)

            # calculate confidence score (= multiply of pred_max_prob)
            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
            confidence_score_list.append(confidence_score)
            # print(pred, gt, pred==gt, confidence_score)

    accuracy = n_correct / float(length_of_data) * 100

    return valid_loss_avg.val(), accuracy, norm_ED, preds_str, confidence_score_list, labels, infer_time, length_of_data


def test(opt):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(opt.device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=opt.device))
    opt.experiment_name = '_'.join(opt.saved_model.split('/')[1:])
    # print(model)

    """ keep evaluation model and result logs """
    os.makedirs(f'./result/{opt.experiment_name}', exist_ok=True)
    os.system(f'cp {opt.saved_model} ./result/{opt.experiment_name}/')

    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(opt.device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(opt.device)  # ignore [GO] token = ignore index 0

    """ evaluation """
    model.eval()
    # with torch.no_grad():
        # if opt.benchmark_all_eval:  # evaluation with 10 benchmark evaluation datasets
        #     benchmark_all_eval(model, criterion, converter, opt)
        # else:
        #     AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        #     eval_data = hierarchical_dataset(root=opt.eval_data, opt=opt)
        #     evaluation_loader = torch.utils.data.DataLoader(
        #         eval_data, batch_size=opt.batch_size,
        #         shuffle=False,
        #         num_workers=int(opt.workers),
        #         collate_fn=AlignCollate_evaluation, pin_memory=True)
        #     _, accuracy_by_best_model, _, _, _, _, _, _ = validation(
        #         model, criterion, evaluation_loader, converter, opt)
        #
        #     print(accuracy_by_best_model)
        #     with open('./result/{0}/log_evaluation.txt'.format(opt.experiment_name), 'a') as log:
        #         log.write(str(accuracy_by_best_model) + '\n')


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
    parser.add_argument('--max_label_length', type=int, default=35, help='the max lenghth of the label')
    parser.add_argument('--epoch', type=int, default=10000, help='number of iterations to train for')
    parser.add_argument('--grad_clip', type=int, default=5, help='grad_clip')
    opt = parser.parse_args()
    opt.num_class = 1556
    test(opt)