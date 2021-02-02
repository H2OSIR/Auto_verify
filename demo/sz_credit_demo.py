# -*- coding: utf-8 -*-
"""深圳信用网的验证码识别 ——— 机器学习入门"""

# step 0: 创建文件夹
import os
os.mkdir('origin_img')
os.mkdir('origin_img_2V')
os.mkdir('train_img')

# step 1-----------------------------------------------------
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

# step 2-----------------------------------------------------
from os import listdir
from PIL import Image

file_list = listdir('origin_img/') 	# 获取图片的文件名（含扩展名），并遍历
for each in tqdm(file_list):
    image = Image.open('origin_img/%s' % each)
    image = process_img.twoValueImage(image, 200)
    image.save('origin_img_2V/%s' % each)
    # 将处理后的图片保存到 origin_img_2V 目录下


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

# step 3-----------------------------------------------------
pass

# step 4-----------------------------------------------------
file_list = listdir('train_img/')
for each in tqdm(file_list):
    img = Image.open('train_img/%s' % each)
    x = process_img.two_Value(img, 'list')  # 生成样本的特征
    y = each[0]  	# 获取样本的标签
    x.insert(0, y)  # 将样本的标签插入到第一列的位置
    process_img.write_csv(fileName='train_csv.csv', values=x)

# step 5-----------------------------------------------------
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