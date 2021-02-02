# -*- coding:utf-8 -*-
"""create in 2017/10/11

    @author: LEE
"""

from . import process_img


class SZ_Captcha:
    """captcha of sz_credit.org"""

    def __init__(self, image):
        """初始化验证码，声明下列属性

        :param image: PIL Image object
        """
        self.image = image      # 图像本身
        self.size = image.size  # 图像尺寸
        self.all_chunks = []    # 所有切块
        self.all_format_chunks = []  # 所有进行重定义尺寸之后的块

    def attributes(self):
        """获取图片的类型和切割线的横坐标位置"""

        if self.size[0] == 90:
            self.type = '11'    # '11'：个位数+个位数
            self.node = (0, 19, 39, 55)
        elif self.size[0] == 120:
            self.type = '22'    # '22'：十位数+十位数
            self.node = (0, 17, 30, 50, 65, 79)
        else:
            self.type = '12'      # '21': 十位数+个位数, '12'：个位数+十位数
            self.node = (0, 19, 39, 53, 67)
            two_value_image = process_img.twoValueImage(image=self.image, G=200)
            for j in range(self.size[1]):
                g = two_value_image.getpixel((40, j))
                # 循环第四十列的颜色，出现黑色则判断为‘21’，否则为‘12’
                if g == 0:
                    self.type = '21'
                    self.node = (0, 17, 30, 50, 67)
                    break

    def crop(self):
        """根据横坐标位置进行切割，将切割后的图片保存到self.all_chunks里。"""

        for i in range(len(self.node) - 1):
            img = self.image.crop((self.node[i], 0, self.node[i + 1], 30))
            self.all_chunks.append(img)
        if self.type == '11' or self.type == '12':
            self.symbol = self.all_chunks[1]
        else:
            self.symbol = self.all_chunks[2]

    def format(self):
        """将self.all_chunks里的图片进行二值、去边框、环切、重定义尺寸，
        并保存至self.all_format_chunks当中"""

        for i, each in enumerate(self.all_chunks):
            two_value_image = process_img.twoValueImage(each, 200)
            remove_frame = process_img.clear_frame(two_value_image, 1)
            cut_around = process_img.cut_around(remove_frame)
            new_img = process_img.format_size(cut_around, (20, 30))
            self.all_format_chunks.append(new_img)

    def recognize(self, model):
        """从self.all_format_chunks中取出图片进行识别，需要传入模型路径。

        :param model str, 由sklearn生成的模型的路径
        :return 四个数字和一个运算符
        """

        result = []
        for each_img in self.all_format_chunks:
            x = process_img.classify(each_img, model=model)
            result.append(x)
        if self.type == '11':
            result.insert(2, 0)
            result.insert(0, 0)
        elif self.type == '12':
            result.insert(0, 0)
        elif self.type == '21':
            result.insert(3, 0)
        else:
            pass
        return result # x1, x2, symbol, x3, x4

    def calculate(self, model):
        """对验证码做数学计算"""

        self.attributes()   # 提取属性
        self.crop()         # 切割
        self.format()       # 格式化切割
        x1, x2, symbol, x3, x4 = self.recognize(model=model)
        if symbol.upper() == 'X':
            result = (int(x1) * 10 + int(x2) * 1) * (int(x3) * 10 + int(x4) * 1)
        else:
            result = (int(x1) * 10 + int(x2) * 1) + (int(x3) * 10 + int(x4) * 1)

        return result
