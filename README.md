# Auto_verify
This object is about automatic verification of captcha in www.szcredit.com

该项目是深圳信用网的验证码的识别，涉及的技术有：爬虫（下载图片）、图像处理（切割验证码）、机器学习分类（识别验证码）。为笔者的第一个机器学习实战项目，在此分享，希望对有需要的人有所帮助。

由于代码写了很久了，有些规范上的问题没有去完善，之后有时间会尽可能地规范下代码，欢迎评论区留言一起完善。

Let's get started! 

## Step 1：下载验证码

通过访问 www.szcredit.com 网站，搜索任意关键字，就能获得该网站的验证码的地址

运行下面的代码，可以很快的下载 500 张随机的验证码，并保存到 `origin_img` 文件夹下：

```python
from tqdm import tqdm # 显示进度条
from sz_credit import process_img

url = 'http://www.szcredit.org.cn/web/WebPages/Member/CheckCode.aspx'
headers = {'user-agent': ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_0) "
                          "AppleWebKit/536.3 (KHTML, like Gecko) "
                          "Chrome/19.0.1063.0 Safari/536.3")}
for i in tqdm(range(0, 500)):
    image = process_img.download_image(url=url, headers=headers)
    with open('origin_img/%s.png' % str(i), 'wb') as f:
        f.write(image)
        f.close()
```

完成后可去对应的目录下查看图片。这一步涉及到的是爬虫的知识，也是最简单的入门级爬虫——爬图片。

下载后的500张图片，分100涨到另一个文件夹，作为后面的测试集用，训练集只用400张图

## Step 2：处理图片

不同的验证码的处理方式不同。这个项目是针对深圳信用网的验证码，由于该网站的验证码比较简单，干扰线与噪点较少，因此处理逻辑也是比较简单，但依然能够有一个很好的识别准确率（后面的模型会验证）。

### 2.1：先将图片处理成二值的

所谓二值图，就是我们看到的是黑白的，其像素点的值为 0 或者 255 。

```python
from os import listdir
from PIL import Image

file_list = listdir('origin_img/') 	# 获取图片的文件名（含扩展名），并遍历
for each in tqdm(file_list):
    image = Image.open('origin_img/%s' % each)
    image = process_img.twoValueImage(image, 200)
    image.save('origin_img_2V/%s' % each)
    # 将处理后的图片保存到 origin_img_2V 目录下
```

### 2.2：对二值图进行切割、处理

这一步需要将黑白的二值图切割成四张图，并提取其中可以用于计算的小图再做格式化处理，处理逻辑会在代码当中记录，需要看源码。

```python
from sz_credit import SZ_Captcha

file_list = listdir('origin_img_2V/')
for each in tqdm(file_list):
    img = Image.open('origin_img_2V/%s' % each)
    img = SZ_Captcha(img)
    img.attributes()  # 提取属性
    img.crop()  # 切割
    img.format()  # 格式化切割
    for i, a in enumerate(img.all_format_chunks):
        a.save('train_img/%s' % (str(i) + each))
```

## Step 3：打标签

这一步需要手动打标签，打标签的方式很简单，就是用文件名的第一个字符表示该图片是什么。如 0.png、0(1).png、+.png、+(1).png。

Images目录里有我已经标注好的样本图片，也可以自己下载，验证代码的完整性。

## Step 4：生成训练集特征

在 step 3 的时候，我们已经标注好了所有的训练集，这一步，需要生成可以用于模型训练的特征。

```python
file_list = listdir('train_img/')
for each in tqdm(file_list):
    img = Image.open('train_img/%s' % each)
    x = process_img.two_Value(img, 'list')  # 生成样本的特征
    y = each[0]  	# 获取样本的标签
    x.insert(0, y)  # 将样本的标签插入到第一列的位置
    process_img.write_csv(fileName='train_csv.csv', values=x)
```

特征的长度是图片像素的长×宽，在进行训练时，为了减少计算量和内存占用，可以去掉样本全为 0 或者全为 1 的特征。

## Step 5：建立模型

```python
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier 
# 使用最简单的 KNN 算法实现

# 加载数据集
data_x, data_y = process_img.loadTrainSet('train_csv.csv')

# 将数据集拆分成训练集和测试集，测试集的比例为 0.3
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.3, random_state=0)

# 实例化 knn 模型
knn = KNeighborsClassifier()

# 拟合(训练)模型
knn.fit(train_x, train_y)

# 训练完成后，对测试集进行预测
test_y_pred = knn.predict(test_x)

# 打印出分类模型在测试集上的评估报告
print(metrics.classification_report(test_y, test_y_pred))

# 打印结果：
              precision    recall  f1-score   support

           +       0.99      1.00      1.00       140
           0       0.96      0.98      0.97        47
           1       0.96      0.80      0.87       135
           2       0.89      0.93      0.91        67
           3       0.86      0.90      0.88        21
           4       0.83      1.00      0.90        19
           5       0.93      1.00      0.96        13
           6       1.00      0.92      0.96        26
           7       0.74      1.00      0.85        14
           8       0.82      1.00      0.90        14
           9       0.81      0.93      0.87        14
           x       0.97      1.00      0.98        85

    accuracy                           0.94       595
   macro avg       0.90      0.96      0.92       595
weighted avg       0.94      0.94      0.94       595
```

从此表，可以看到模型的精确度、召回率、f1-score三个评价指标，support是指属于该类别的样本个数。

这样，一个验证码识别的项目就结束了，期间涉及到的技术我这里也只是简单带过，更详细的可以参考源代码。