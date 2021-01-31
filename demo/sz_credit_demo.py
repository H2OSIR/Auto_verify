# -*- coding: utf-8 -*-
"""深圳信用网的验证码识别 ——— 机器学习入门"""

import numpy as np
import pandas as pd
from os import listdir
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

from sz_credit import process_img
import sz_credit.sz_credit as sz_credit


# 对原始图片做预处理
origin_img_path = '../Images/origin_img'
file_list = listdir('origin_img_path')
i = 0
for each in file_list:
    image = Image.open(f'{origin_img_path}/{each}')

    # 1. 二值化成黑白图
    new_image = process_img.twoValueImage(image, 200)

    # 二值图可保存可不保存
    # new_image.save('F:/GitHub/Auto_verify/Images/origin_img_2V/%s' % each)

    # 2. 将验证码切割成四块，实例化具体的方法
    img = sz_credit.SZ_Captcha(image=new_image)

    # 获取对象属性
    img.attributes()

    # 3. 切割图片 并保存每块图
    img.crop()

    # 4. 重定义所有切块的尺寸
    img.format()

    for eachs in img.all_format_chunks:
        i += 1
        # 5. 保存切割好的每一块图到指定文件夹，之后手动进行打标签。
        eachs.save('../Images/train_image/%s' % str(i) + '.png')

# 6. 生成 csv 训练集
train = pd.read_csv('D:\\DataAnalyse\\PyDataAnalyse\\projects\\szcredit_captcha\\train_data.csv')
trains = train.iloc[:, 1:]
labels = train.iloc[:, 0]

# 拆分训练集
train_data, test_data, train_target, test_target = train_test_split(trains, labels, test_size=0.3, random_state=0)

# 建立模型
clf = DecisionTreeClassifier()

# 训练模型
clf.fit(train_data, train_target)

# 预测测试集 并评估模型
pred_test = clf.predict(test_data)

# 输出分类报告
print(classification_report(test_target, pred_test))

# 对于新的验证码，要进行与训练集图片同样的方式处理

new_image = Image.open('test_image/0.png')
# image.show()
new_image = process_img.twoValueImage(new_image, 200)
# 实例化图对象
img = sz_credit.SZ_Captcha(image=new_image)

# 获取对象属性
img.attributes()

# 切割图片 并保存每块图
img.crop()

# 重定义所有切块的尺寸
img.format()

# 识别每个切块，并插入占位的 0
x1, x2, symbol, x3, x4 = img.recognize(model=clf)

if symbol.upper() == 'X':
    result = (int(x1) * 10 + int(x2) * 1) * (int(x3) * 10 + int(x4) * 1)
else:
    result = (int(x1) * 10 + int(x2) * 1) + (int(x3) * 10 + int(x4) * 1)

print(result)

print(clf)