import os
import glob
import cv2
import argparse

def get_dict(dict_path):
    '''
    生成字典
    :return: tuple(dict,dict)
    '''
    chars_ptr = open(dict_path, "r", encoding='utf-8')
    chars = chars_ptr.read().splitlines()
    chars_ptr.close()

    char2num_dict={}
    num2char_dict={}

    for i,char in enumerate(chars):
        char2num_dict[char]=i
        num2char_dict[i]=char

    return (char2num_dict,num2char_dict)




def make_txt(opt):
    '''
    :param num_txt_path:生成训练集txt 标签是数字
    :param img_dir:图片路径
    :param text_txt_path:生成训练集txt 标签是文本
    :param dict_path:字典路径
    图片名称为图片文字标签+.jpg
    :return:
    '''
    dict_path=opt.dict_path
    img_dir=opt.img_dir
    num_txt_path=opt.num_txt_path
    text_txt_path=opt.text_txt_path

    dicts=get_dict(dict_path)
    char2num_dict=dicts[0]
    num2char_dict=dicts[1]

    jpgfile = glob.glob(pathname=os.path.join(img_dir,'*.jpg'))
    length = len(jpgfile)
    num_txt=open(num_txt_path,'w')
    text_txt=open(text_txt_path,'w')
    for filename in jpgfile:
        label=os.path.split(filename)[1].rstrip()
        label=os.path.splitext(label)[0].rstrip() #中文label
        img_path=os.path.join(img_dir,filename)
        num_txt.write(img_path)
        text_txt.write(img_path)
        text_txt.write(' ')
        text_txt.write(label)
        text_txt.write('\n')
        for char in label:
            num_txt.write(' ')
            num_txt.write(str(char2num_dict[char]))
        num_txt.write('\n')

        '''去掉无法读取的图片'''
        img=cv2.imread(img_path)
        if img is None:
            continue

        '''去掉过长标签等'''


    print('nums of images',length)
    num_txt.close()
    text_txt.close()



if __name__=='__main__':
    #test
    parser=argparse.ArgumentParser()
    parser.add_argument('--dict_path', type=str, default=r'/home/luoyc/zhulin/textDR/utils/dict.txt', help='dict path')
    parser.add_argument('--img_dir', type=str, default=r'/home/luoyc/zhulin/img_crop', help='the image folder dir')
    parser.add_argument('--num_txt_path', type=str, default=r'/home/luoyc/zhulin/textDR/utils/num_train.txt',
                        help='num_txt_path')
    parser.add_argument('--text_txt_path', type=str, default=r'/home/luoyc/zhulin/textDR/utils/text_train.txt',
                        help='text_txt_path')
    opt = parser.parse_args()
    make_txt(opt)





