v_0
### 数据准备
训练图片和测试图片放在同一个文件夹中，另 图片名称是图片文字标签
通过设置config.py中图片路径，num_txt,text_txt路径
命令行运行以下命令
```
python make_txt_dict.py 
```
可以生成以下格式的txt
text_txt:
./.../图片名称.jpg 文字标签

num_txt:（转换成字典中索引标签）

./.../图片名称.jpg 26 78 78 45 ...

### 模型
FeatureExtraction:vgg/resNet
SequenceModeling:biLSTM
Prediction:CTC/Attention

### 训练
命令行运行以下命令修改模型配置
```
python train.py --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn
```

### 测试

