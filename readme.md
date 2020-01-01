v_0
### 数据准备
命令行运行以下命令
```
python make_txt_dict.py --dict_path dict_path --img_dir img_dir --num_txt_path num_txt_path /
--text_txt_path text_txt_path
```
可以生成以下格式的txt
- text_txt:

./.../图片名称.jpg 文字标签

- num_txt:（转换成字典中索引标签）

./.../图片名称.jpg 26 78 78 45 ...

### 数据装载
dataset.py 
为了使attn和ctc两种方式利用统一接口编码解码，使用文字标签
```
 self.img_label_list=self.read_text_file(opt.text_txt_path)
```
若想生成batch里是数字标签 ，请将以上代码修改为
```
self.img_label_list=self.read_num_file(opt.num_txt_path)
```
### 模型
- FeatureExtraction:vgg/resNet
- SequenceModeling:biLSTM
- Prediction:CTC/Attention

### 训练
命令行运行以下命令修改模型配置
```
python train.py --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC
```

ps:预测方式选择attn时出现CUBLAS_STATUS_INTERNAL_ERROR
参考['cublas runtime error' for (not so large) *fp16* matrix multiplication #24018](https://github.com/pytorch/pytorch/issues/24018)
需要cuda10.1 release 版本 但是现在该版本并没有出来 


