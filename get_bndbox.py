import cv2
import numpy as np
from collections import OrderedDict
from sklearn.cluster import DBSCAN
import pytesseract

tesseract_cmd = r'D:\programfiles\tesseract\tesseract'
pytesseract.pytesseract.tesseract_cmd = tesseract_cmd


def remove_duplicates(rectangles):
    # 将列表中的每个元素（列表）转换为元组，使用OrderedDict来保持顺序并删除重复元素
    unique_rectangles = list(OrderedDict.fromkeys(tuple(rect) for rect in rectangles))
    # 将元组转换回列表
    unique_rectangles = [list(rect) for rect in unique_rectangles]
    return unique_rectangles


def red_region_detection(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义HSV中红色的范围
    # OpenCV中HSV范围: H: 0-180, S: 0-255, V: 0-255
    # 红色的H值大约在0度附近以及360度附近，需要两个范围覆盖环绕的特性
    lower_red1 = np.array([0, 70, 150])
    upper_red1 = np.array([13, 255, 255])
    lower_red2 = np.array([165, 70, 100])
    upper_red2 = np.array([180, 255, 255])

    # 根据定义的HSV范围创建掩码
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # 将掩码应用到原图像，提取红色区域
    red_regions = cv2.bitwise_and(image, image, mask=red_mask)
    return red_regions


def amber_region_detection(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 定义HSV中琥珀色的范围
    # 琥珀色通常可以被视为介于橙色和黄色之间的颜色
    # 通常在HSV空间中，琥珀色的色调（Hue）大约在20到30度
    lower_amber = np.array([19, 50, 50])  # H范围稍宽，S和V根据实际情况调整
    upper_amber = np.array([30, 220, 255])
    # 根据定义的HSV范围创建掩码
    amber_mask = cv2.inRange(hsv_image, lower_amber, upper_amber)
    # 将掩码应用到原图像，提取琥珀色区域
    amber_regions = cv2.bitwise_and(image, image, mask=amber_mask)
    # amber_regions_close = cv2.bitwise_and(image, image, mask=cv2.erode(cv2.dilate(amber_mask, k),k))
    return amber_regions


def get_rect_points(text_boxes):
    x1 = np.min(text_boxes[:, 0])
    y1 = np.min(text_boxes[:, 1])
    x2 = np.max(text_boxes[:, 2])
    y2 = np.max(text_boxes[:, 3])
    return [x1, y1, x2, y2]


def calculate_IoU(rect1, rect2, type='min'):
    """
    computing the IoU of two boxes.
    Args:
        box: (x1, y1, x2, y2),通过左上和右下两个顶点坐标来确定矩形
    Return:
        IoU: IoU of box1 and box2.
    """
    px1 = rect1[0]
    py1 = rect1[1]
    px2 = rect1[2]
    py2 = rect1[3]

    gx1 = rect2[0]
    gy1 = rect2[1]
    gx2 = rect2[2]
    gy2 = rect2[3]

    # Check if rect1 contains rect2
    if px1 <= gx1 and px2 >= gx2 and py1 <= gy1 and py2 >= gy2:
        return 1

    # Check if rect2 contains rect1
    if gx1 <= px1 and gx2 >= px2 and gy1 <= py1 and gy2 >= py2:
        return 1

    parea = (px2 - px1) * (py2 - py1)  # 计算P的面积
    garea = (gx2 - gx1) * (gy2 - gy1)  # 计算G的面积

    # 求相交矩形的左上和右下顶点坐标(x1, y1, x2, y2)
    x1 = max(px1, gx1)  # 得到左上顶点的横坐标
    y1 = min(py1, gy1)  # 得到左上顶点的纵坐标
    x2 = min(px2, gx2)  # 得到右下顶点的横坐标
    y2 = max(py2, gy2)  # 得到右下顶点的纵坐标

    # 利用max()方法处理两个矩形没有交集的情况,当没有交集时,w或者h取0,比较巧妙的处理方法
    # w = max(0, (x2 - x1))  # 相交矩形的长，这里用w来表示
    # h = max(0, (y1 - y2))  # 相交矩形的宽，这里用h来表示
    # print("相交矩形的长是：{}，宽是：{}".format(w, h))
    # 这里也可以考虑引入if判断
    w = x2 - x1
    h = y1 - y2
    if w <= 0 or h <= 0:
        return 0
    area = w * h  # G∩P的面积
    # 并集的面积 = 两个矩形面积 - 交集面积
    # IoU = area / (parea + garea - area)
    if type == 'min':
        IoU = area / (min(parea, garea))
    else:
        IoU = area / (parea + garea - area)

    return IoU


class BoxesConnector(object):
    def __init__(self, rects, imageW, max_dist=None, overlap_threshold=None):
        self.rects = np.array(rects)
        self.imageW = imageW
        self.max_dist = max_dist  # x轴方向上合并框阈值
        self.overlap_threshold = overlap_threshold  # y轴方向上最大重合度
        self.graph = np.zeros((self.rects.shape[0], self.rects.shape[0]))  # 构建一个N*N的图 N等于rects的数量

        self.r_index = [[] for _ in range(imageW)]  # 构建imageW个空列表
        for index, rect in enumerate(rects):  # r_index第rect[0]个元素表示 第index个(数量可以是0/1/大于1)rect的x轴起始坐标等于rect[0]
            if int(rect[0]) < imageW:
                self.r_index[int(rect[0])].append(index)
            else:  # 边缘的框旋转后可能坐标越界
                self.r_index[imageW - 1].append(index)
        # print(self.r_index)

    def calc_overlap_for_Yaxis(self, index1, index2):
        # 计算两个框在Y轴方向的重合度(Y轴错位程度)
        height1 = self.rects[index1][3] - self.rects[index1][1]
        height2 = self.rects[index2][3] - self.rects[index2][1]
        y0 = max(self.rects[index1][1], self.rects[index2][1])
        y1 = min(self.rects[index1][3], self.rects[index2][3])
        Yaxis_overlap = max(0, y1 - y0) / (0.5 * (height1 + height2))
        return Yaxis_overlap

    def calc_IOU(self, index1, index2):
        # 计算两个框在Y轴方向的重合度(Y轴错位程度)
        rect1 = self.rects[index1]
        rect2 = self.rects[index2]
        iou = calculate_IoU(rect1, rect2, 'min')
        return iou

    def get_proposal(self, index):
        rect = self.rects[index]
        height = rect[3] - rect[1]
        for left in range(rect[0] + 1, min(self.imageW - 1, rect[2] + self.max_dist)):
            # print('left',left)
            for idx in self.r_index[left]:
                # index: 第index个rect(被比较rect)
                # idx: 第idx个rect的x轴起始坐标大于被比较rect的x轴起始坐标(+max_dist)且小于被比较rect的x轴终点坐标(+max_dist)
                if self.calc_overlap_for_Yaxis(index, idx) > self.overlap_threshold:
                    return idx
                # elif self.calc_IOU(index, idx) > 0.85:
                #    return idx

        return -1

    def sub_graphs_connected(self):
        sub_graphs = []  # 相当于一个堆栈
        for index in range(self.graph.shape[0]):
            # 第index列全为0且第index行存在非0
            if not self.graph[:, index].any() and self.graph[index, :].any():  # 优先级是not > and > or
                v = index
                sub_graphs.append([v])
                # 级联多个框(大于等于2个)
                while self.graph[v, :].any():
                    v = np.where(self.graph[v, :])[0][
                        0]  # np.where(self.graph[v, :])：(array([5], dtype=int64),)  np.where(self.graph[v, :])[0]：[5]
                    sub_graphs[-1].append(v)
        return sub_graphs

    def connect_boxes(self):
        for idx, _ in enumerate(self.rects):
            proposal = self.get_proposal(idx)
            if proposal >= 0:
                self.graph[idx][proposal] = 1  # 第idx和proposal个框需要合并则置1

        sub_graphs = self.sub_graphs_connected()  # sub_graphs [[0, 1], [3, 4, 5]]

        # 不参与合并的框单独存放一个子list
        set_element = set([y for x in sub_graphs for y in x])  # {0, 1, 3, 4, 5}
        for idx, _ in enumerate(self.rects):
            if idx not in set_element:
                sub_graphs.append([idx])  # [[0, 1], [3, 4, 5], [2]]

        result_rects = []
        for sub_graph in sub_graphs:
            rect_set = self.rects[list(sub_graph)]  # [[228  78 238 128],[240  78 258 128]].....
            rect_set = get_rect_points(rect_set)
            result_rects.append(rect_set)
        return np.array(result_rects)


def remove_contained_boxes(bboxes):
    # Function to check if one box contains another
    def contains(box1, box2):
        # Unpack the coordinates
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        return x1 <= x3 and x2 >= x4 and y1 <= y3 and y2 >= y4

    # Initialize list to store boxes to remove
    to_remove = set()

    # Compare each box with every other box
    for i in range(len(bboxes)):
        for j in range(len(bboxes)):
            if i != j and contains(bboxes[i], bboxes[j]):
                to_remove.add(j)  # Add the index of the contained box to the removal list
        # print((bboxes[i][2]-bboxes[i][0])/(bboxes[i][3]-bboxes[i][1]))
        if ((bboxes[i][2] - bboxes[i][0]) / (bboxes[i][3] - bboxes[i][1])) < 3:
            to_remove.add(i)
        # Create a new list without the contained boxes
    remaining_boxes = [bboxes[i] for i in range(len(bboxes)) if i not in to_remove]
    return remaining_boxes


def combine_insert_boxes(bboxes):
    th = 0.1
    dlt_list = []
    cb_list = []
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            if calculate_IoU(bboxes[i], bboxes[j]) > th:
                dlt_list.append(bboxes[i])
                dlt_list.append(bboxes[j])
                cb_list.append(
                    [min(bboxes[i][0], bboxes[j][0]), min(bboxes[i][1], bboxes[j][1]), max(bboxes[i][2], bboxes[j][2]),
                     max(bboxes[i][3], bboxes[j][3])])
    if len(cb_list) > 0:
        filtered_boxes = [item for item in bboxes if item not in dlt_list]
        unique_rectangles = []
        for rect in cb_list:
            if rect not in unique_rectangles:
                unique_rectangles.append(rect)
        filtered_boxes = filtered_boxes + unique_rectangles

    else:
        filtered_boxes = bboxes
    return filtered_boxes


def method_3(image):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    t, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary


def calculate_variance(image, rect):
    x1, y1, x2, y2 = rect
    roi = image[y1:y2, x1:x2]
    roi = roi.reshape(-1, 3)
    variance = np.var(roi, axis=0)
    return variance


def calculate_diff(image, rect):
    x1, y1, x2, y2 = rect
    roi = image[y1:y2, x1:x2]
    roi = roi.reshape(-1, 3)
    max_value = np.max(roi, axis=0)
    min_value = np.min(roi, axis=0)
    return max_value - min_value


def find_outliers(bboxes):  # 提取所有矩形的横坐标
    x_coords = [rect["xmin"] for rect in bboxes]
    x_coords2 = [rect["xmax"] for rect in bboxes]
    x_coords3 = [rect["ymax"] - rect["ymin"] for rect in bboxes]
    # 计算四分位数
    '''
    Q1 = np.percentile(x_coords, 25)
    Q3 = np.percentile(x_coords, 75)
    IQR = Q3 - Q1
    # 计算离群值的界限
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # 找到离群的矩形
    outliers = [i for i, rect in enumerate(bboxes) if rect["xmin"] < lower_bound or rect["xmin"] > upper_bound]
    '''
    # 将数据转换为二维数组形式，因为DBSCAN需要多维输入
    data = np.array(x_coords).reshape(-1, 1)
    data2 = np.array(x_coords2).reshape(-1, 1)
    data3 = np.array(x_coords3).reshape(-1, 1)
    # 使用DBSCAN进行聚类
    db = DBSCAN(eps=50, min_samples=2).fit(data)
    db2 = DBSCAN(eps=50, min_samples=2).fit(data2)
    db3 = DBSCAN(eps=50, min_samples=2).fit(data3)
    # 获取聚类标签
    labels = db.labels_
    labels2 = db2.labels_
    # labels3 = db3.labels_
    # 标记离群点
    remain_indices = [i for i, label in enumerate(labels) if label != -1]
    remain_indices2 = [j for j, label in enumerate(labels2) if label != -1]
    # remain_indices3 = [j for j, label in enumerate(labels3) if label != -1]
    merged_list = list(set(remain_indices + remain_indices2))
    # print("不离群的矩形:", merged_list)

    return merged_list


def Morph_exam(test_img, binary):
    # 形态核：膨胀让轮廓突出--- 腐蚀去掉细节--再膨胀，让轮廓更明显
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 7))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilation = (cv2.dilate(binary, element1, iterations=2))
    # cv2.morphologyEx(binary, cv2.MORPH_CLOSE, element1, iterations=3)) # 闭运算
    # dilation = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, element2)  # 开运算

    h, w, c = test_img.shape
    test_img_cp1 = test_img.copy()
    test_img_cp2 = test_img.copy()
    # 查找轮廓和筛选文字区域
    region = []
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cnt = contours[i]
        rect = cv2.minAreaRect(cnt)
        # 获取box四点坐标, 根据文字特征，筛选可能是文本区域的矩形。
        box = cv2.boxPoints(rect)
        rect1 = cv2.boundingRect(cnt)

        if rect1[2] > 1500 or rect1[3] > 200:
            continue
        if (rect1[2]) > w * 2 / 3:
            continue
        if rect1[0] > w * 4 / 7:
            continue
        if rect1[0] + rect1[2] > w * 3 / 4:
            continue
        region.append([rect1[0], rect1[1], rect1[0] + rect1[2], rect1[1] + rect1[3]])

    # 合并接近的矩形
    new_rects1 = []
    new_rects2 = region.copy()
    i = 1
    while len(new_rects2) != (len(new_rects1)):
        new_rects1 = new_rects2
        connector = BoxesConnector(new_rects1, w, max_dist=int(80 / i), overlap_threshold=0.75 / i)
        new_rects11 = connector.connect_boxes()
        new_rects2 = combine_insert_boxes(new_rects11.tolist())  # combine_insert_boxes
        # print(str(len(new_rects2)) + '\n')
        i = i + 1

    # 绘制轮廓
    new_rects = new_rects2.copy()
    n = 0
    filtered_boxes_cp = new_rects  # new_rects
    i = 0
    removeList = []
    for i in range(len(filtered_boxes_cp)):
        rect = filtered_boxes_cp[i]
        n = n + 1
        if rect[0] < 0 or rect[0] < 0:
            removeList.append(rect)
            continue
        if (rect[3] - rect[1]) not in range(20, 150) or (rect[2] - rect[0]) not in range(80, 1000):
            removeList.append(rect)
            continue
        if (rect[0] not in range(0, int(w / 2))) or ((rect[1]) not in range(int(500), int(h))):
            removeList.append(rect)
            continue

    filtered_boxes = [item for item in filtered_boxes_cp if item not in removeList]
    return filtered_boxes


def word_extract(img):
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 6)
    # 形态核：膨胀让轮廓突出--- 腐蚀去掉细节--再膨胀，让轮廓更明显
    element1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    element2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 5))  #
    # erosion = cv2.erode(binary, element1, iterations=1)
    dilation2 = cv2.dilate(binary, element2, iterations=3)
    contours, hierarchy = cv2.findContours(dilation2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    out_img = np.zeros((dilation2.shape[0], dilation2.shape[1]))
    '''
    for i in range(len(contours)):
        cnt = contours[i]
        # 通过轮廓填充。
        cv2.fillPoly(out_img, [cnt], color=255)
    cv2.namedWindow("cnt", 0)
    cv2.imshow("cnt", out_img)
    cv2.waitKey(0)'''
    return dilation2


def red_bnd_extract(img, word_img):
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    red_region = red_region_detection(cv_img)
    gray = cv2.cvtColor(red_region, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # 高斯滤波去除小噪点
    # 使用二值化将图像转换为仅包含黑白的图形
    t, red_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    red_binary = cv2.bitwise_and(red_binary, word_img.astype(np.uint8))
    red_boxes = Morph_exam(red_region, red_binary)
    return red_boxes


def amber_bnd_extract(img, word_img):
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    # amber_binary0 = cv2.adaptiveThreshold(hsv[:,:,0], 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
    #                            cv2.THRESH_BINARY_INV, 11, 2)
    # amber_binary0 = cv2.adaptiveThreshold(hsv[:, :, 1], 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # element2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    # amber_binary0= cv2.dilate(amber_binary0, element2, iterations=1)

    amber_region = amber_region_detection(cv_img)
    gray = cv2.cvtColor(amber_region, cv2.COLOR_BGR2GRAY)
    # gray = cv2.bilateralFilter(gray, 9, 150, 150)
    # 使用二值化将图像转换为仅包含黑白的图形
    t, amber_binary1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY |
                                     cv2.THRESH_OTSU)
    # amber_binary2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
    # cv2.THRESH_BINARY_INV, 11, 2)
    element2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    amber_binary2 = cv2.dilate(amber_binary1, element2, iterations=3)

    # amber_binary = cv2.bitwise_and(amber_binary0, amber_binary1)
    amber_binary = cv2.bitwise_and(amber_binary1, word_img.astype(np.uint8))
    amber_boxes = Morph_exam(amber_region, amber_binary)
    # new_connector = BoxesConnector(filtered_boxes, w, max_dist=35, overlap_threshold=0.25)
    # filtered_boxes2 = new_connector.connect_boxes()
    # filtered_boxes=filtered_boxes2.tolist()
    return amber_boxes


def rectangles_near(rect1, rect2, th=0.2):
    x0_1, y0_1, x1_1, y1_1 = rect1
    x0_2, y0_2, x1_2, y1_2 = rect2
    # print(y1_2-y0_1,y1_1-y0_2)
    if th == 0:
        if (x0_1 >= x0_2 and x1_1 <= x1_2) and (y0_1 >= y0_2 and y1_1 <= y1_2):
            return 1
        elif (x0_2 >= x0_1 and x1_2 <= x1_1) and (y0_2 >= y0_1 and y1_2 <= y1_1):
            return 1
        else:
            return 0

    else:
        if (x0_2 >= x0_1 and x0_2 <= x1_1) and (y0_2 >= y0_1 and y0_2 <= y1_1):
            return 1
        elif (x0_1 >= x0_2 and x0_1 <= x1_2) and (y0_1 >= y0_2 and y0_1 <= y1_2):
            return 1
        elif (x0_1 >= x0_2 and x0_1 <= (x1_2 + 10)) and (y0_1 >= y0_2 and y0_1 <= y1_2):
            return 1
        elif (x0_2 >= x0_1 and x0_2 <= x1_1) and abs(y1_1 - y0_2) < min(abs(th * (y1_1 - y0_1)),
                                                                        abs(th * (y1_2 - y0_2))):
            return 1
        elif (x0_1 >= x0_2 and x0_1 <= x1_2) and abs(y1_2 - y0_1) < min(abs(th * (y1_1 - y0_1)),
                                                                        abs(th * (y1_2 - y0_2))):
            return 1

        else:
            return 0


def merge_rectangles(rect1, rect2):
    x0_1, y0_1, x1_1, y1_1 = rect1
    x0_2, y0_2, x1_2, y1_2 = rect2
    x0 = min(x0_1, x0_2)
    y0 = min(y0_1, y0_2)
    x1 = max(x1_1, x1_2)
    y1 = max(y1_1, y1_2)
    return [x0, y0, x1, y1]


def enlarge_rectangle(rect_ls, scale_factor=1.1):
    enlarged_ls = []

    for i in range(len(rect_ls)):
        # 提取原始坐标
        x0, y0, x1, y1 = rect_ls[i]

        # 计算中心点坐标
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2

        # 计算原始长和宽
        width = x1 - x0
        height = y1 - y0

        # 放大长和宽
        new_width = width * scale_factor
        new_height = height * scale_factor

        # 计算新的矩形坐标
        x01 = max(0, int(center_x - new_width / 2))
        y01 = max(0, int(center_y - new_height / 2))
        x11 = int(center_x + new_width / 2)
        y11 = int(center_y + new_height / 2)
        enlarged_rect = [x01, y01, x11, y11]
        enlarged_ls.append(enlarged_rect)

    return enlarged_ls


def cal_area(box):
    area = (box[2] - box[0]) * (box[3] - box[1])
    return area


def merge_near_rectangles(rectangles, th=0.2):
    merged = []
    row = 0
    while rectangles:
        rect = rectangles.pop(0)
        has_merged = False
        if (cal_area(rect) > 150000):
            merged.append(rect)
            row = row + 1
            continue
        for i in range(len(rectangles)):
            fg = (rectangles_near(rect, rectangles[i]))
            gg = calculate_IoU(rect, rectangles[i], type='min') > th
            if fg > 0 or gg:
                merged_final_rect = merge_rectangles(rect, rectangles[i])
                merged.append(merged_final_rect)
                row = row + 2 * fg
                has_merged = True
                rectangles.pop(i)
                break
        if not has_merged:
            merged.append(rect)
            row = row + 1
    return merged, row


'''
    # 再次检查已合并的矩形列表，确保没有漏掉相交的矩形
    merged_final = []
    merged_r = merged[::-1]
    while merged_r:
        rect = merged_r.pop(0)
        has_merged = False
        for i in range(len(merged_r)):
            if rectangles_near(rect, merged_r[i]):
                merged_final_rect = merge_rectangles(rect, merged_r[i])
                merged_final.append(merged_final_rect)
                has_merged = True
                merged_r.pop(i)
                break
        if not has_merged:
            merged_final.append(rect)
    merged_final = enlarge_rectangle(merged_final)
    '''


def calculate_entropy(roi):
    # 计算灰度直方图
    hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
    hist = hist / hist.sum()

    # 计算熵
    entropy = -np.sum(hist * np.log2(hist + 1e-7))  # 加1e-7以避免log(0)
    return entropy


def contains_text_ls(image, bboxes):
    flag_ls = []
    th = 10
    edge_density_ls = []
    for box in bboxes:
        roi = image[box[1]:box[3], box[0]:box[2]]
        edges = cv2.Canny(roi, 50, 150)
        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
        if edge_density > th:
            flag_ls.append(1)
        else:
            flag_ls.append(0)
        edge_density_ls.append(edge_density)
    return flag_ls, edge_density_ls


def has_edges(image, threshold=100):
    edges = cv2.Canny(image, 50, 150)
    edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
    return edge_density > threshold


def get_boxes(img):
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    h, w, c = cv_img.shape
    ret, word_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # word_extract(img)
    '''
    contours1, hierarchy1 = cv2.findContours(word_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    element1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 3))
    dilation = cv2.dilate(word_bin, element1, iterations=1)
    element2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 5))
    erosion = cv2.erode(dilation, element2, iterations=1)
    contours2, hierarchy2 = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.namedWindow("word_erosion",0)
    cv2.imshow("word_erosion",erosion)

    word_bin=erosion.copy()

    mask = np.zeros((h, w), dtype=np.uint8)
    filtered_contours = [cnt for cnt in contours1 if (cv2.contourArea(cnt) >0 and cv2.contourArea(cnt) <1500) ]#<1500
    cv2.drawContours(mask, filtered_contours, -1, 255, -1)
    filtered_contours2 = [cnt for cnt in contours2 if (cv2.contourArea(cnt) > 0 and cv2.contourArea(cnt) < 4000)]
    cv2.drawContours(mask, filtered_contours2, -1, 255, -1)
    word_bin=cv2.bitwise_and(mask,word_bin)'''
    # cv2.namedWindow("word_bin",0)
    # cv2.waitKey(0)
    amber_boxes = amber_bnd_extract(img, word_bin)
    red_boxes = red_bnd_extract(img, word_bin)
    red_conf = 0
    top = 0
    left = 0
    bottom = 0
    right = 0
    row_r = 0
    if len(red_boxes) > 0:
        red_boxes, row_r = merge_near_rectangles(red_boxes)
        if min([red_box[3] for red_box in red_boxes]) > h / 2 and max(
                [red_box[3] - red_box[1] for red_box in red_boxes]) > 140:
            row_r = 3 * row_r
        if min([red_box[3] for red_box in red_boxes]) < h / 2 and max(
                [red_box[3] - red_box[1] for red_box in red_boxes]) > 95:
            row_r = 3 * row_r
        red_conf = 1
        top = min([red_box[1] for red_box in red_boxes]) - 10
        left = min([red_box[0] for red_box in red_boxes]) - 10
        bottom = top + 8 * sum([red_box[3] - red_box[1] for red_box in red_boxes]) / max(len(red_boxes), row_r)
        right = max(left + 12 * sum([red_box[3] - red_box[1] for red_box in red_boxes]) / max(len(red_boxes), row_r),
                    max([red_box[2] for red_box in red_boxes]))
    amber_boxes, row_a = merge_near_rectangles(amber_boxes)
    boxes = amber_boxes + red_boxes
    row = row_a + row_r

    boxes, row1 = merge_near_rectangles(boxes)
    # boxes = (amber_boxes) + (red_boxes)
    # entropy_ls = calculate_entropy_ls(gray,boxes)
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    bnd_ls = []
    h_ls = []
    sorted_rectangles = sorted(boxes, key=lambda rect: rect[1])
    boxes = sorted_rectangles
    edged_ls, edge_density_ls = contains_text_ls(gray, sorted_rectangles)
    edge_density_rls = []
    for i in range(len(sorted_rectangles)):
        box = {"xmin": boxes[i][0], "ymin": boxes[i][1], "xmax": boxes[i][2], "ymax": boxes[i][3]}
        if (boxes[i][0] < 0 or boxes[i][1] < 0 or boxes[i][2] > w or boxes[i][3] > h):
            continue
        if (boxes[i][2] - boxes[i][0] <= 200):
            continue
        if (boxes[i][1] <= h / 2 or boxes[i][2] >= 2 * w / 3):
            continue
        if edged_ls[i] == 0:
            continue
        if (boxes[i][2] - w / 2) > 0 and (boxes[i][2] - w / 2) > abs(boxes[i][0] - w / 2):
            continue
        h_ls.append(boxes[i][3] - boxes[i][1])
        bnd_ls.append(box)
        edge_density_rls.append(edge_density_ls[i])
    if len(bnd_ls) > 2:
        remaining_indices = find_outliers(bnd_ls)
        if len(remaining_indices) > 0:
            if remaining_indices[0] != 0 and np.argmax(edge_density_rls) == 0:
                remaining_indices.append(0)
            remain_bnd_ls = [bnd_ls[i] for i in remaining_indices]
        else:
            remain_bnd_ls = bnd_ls
    else:
        remain_bnd_ls = bnd_ls
    warning_ls = []
    if len(remain_bnd_ls) > 0:
        min_xmin = max(0, min(remain_bnd_ls, key=lambda b: b["xmin"])["xmin"])
        min_ymin = max(0, min(remain_bnd_ls, key=lambda b: b["ymin"])["ymin"])
        max_xmax = max(remain_bnd_ls, key=lambda b: b["xmax"])["xmax"]
        max_ymax = max(remain_bnd_ls, key=lambda b: b["ymax"])["ymax"]

        roi_value = {"xmin": max(0, int(0.95 * (min_xmin) / (1)) - 8),
                     "ymin": int(0.95 * (min_ymin) / (+1)) - 8,
                     "xmax": min(w, int(1.05 * (max_xmax) / (1)) + 8),
                     "ymax": min(h,
                                 int((min(min_ymin + min(8 * sum(h_ls) / len(h_ls), 2 * h / 7), max_ymax)) / (1)) + 8)}
        for i in range(len(remain_bnd_ls)):
            # for bnd in remain_bnd_ls:
            #    warning = {"name": "warning", "bndbox": bnd}
            #    warning_ls.append(warning)

            rect = remain_bnd_ls[i]
            # cv2.rectangle(cv_img,(max(0,int(0.5*rect["xmin"]+0.5*min_xmin)-5), rect["ymin"]-5), (rect["xmax"], rect["ymax"]),(0, 255, 0),2)
            warning = {"xmin": max(0, int(0.5 * rect["xmin"] + 0.5 * min_xmin) - 5),
                       "ymin": rect["ymin"] - 5, "xmax": rect["xmax"],"ymax": rect["ymax"]}
            warning_ls.append(warning)
    else:
        roi_value = {}
    roi = {"name": "roi", "bndbox": roi_value}

    return warning_ls, roi_value