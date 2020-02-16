import cv2
import os
import numpy as np
from pascal_voc_writer import Writer


# 函数1 用来检测mask的contour
def contour_detect(cv2image):
    gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(cv2image, contours, -1, (0, 0, 255), 2)
    return contours, hierarchy


# 函数2 根据bbox坐标产生矩形的mask
def rectangle_mask(box, cv2image):
    white = (255, 255, 255)
    for i, j in enumerate(box):  # [xmin ymin xmax ymax] x对应的是W，y对应的是H
        cv2image[j[1]:j[3], j[0]:j[2], :] = white
    return


# 函数3 将bbox和图片信息等产生一个pascal voc 2007格式的xml标注文件
def pascal_xml_writer(imgname, cv2image, corrds, output_dir):
    h, w, _ = cv2image.shape
    writer = Writer(imgname, w, h)
    for corrd in corrds:
        writer.addObject('polyp', corrd[0], corrd[1], corrd[2], corrd[3])
    writer.save(os.path.join(output_dir, imgname.split('.')[0] + '.xml'))
    return


# 函数4 cv2显示图片函数
def cv2show(cv2image, window_name):
    cv2.imshow(window_name, cv2image)
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyWindow(window_name)
    else:
        raise RuntimeError('undefined key value, "esc" only.')


# 函数5 从contours和hierarchy中解出 四个方向的定点 [(xmin, ymin, xmax, ymax), ...]
def pts_extract(contours):
    pts = []
    for i in range(len(contours)):
        xmin = contours[i][:, :, 0].min()
        ymin = contours[i][:, :, 1].min()
        xmax = contours[i][:, :, 0].max()
        ymax = contours[i][:, :, 1].max()

        if xmax - xmin < 10 or ymax - ymin < 10:
            pass
        else:
            pts.append([xmin, ymin, xmax, ymax])
    return pts


# 函数6 将矩形框画在原图上
def drawbox(pts, img):
    for i in range(len(pts)):
        cv2.rectangle(img, tuple(pts[i][0:2]), tuple(pts[i][2:]), (255, 0, 0), 2)


if __name__ == '__main__':
    imgdir = '/Users/jiaxin/MICCAI2020/CVC-EndoSceneStill/CVC-300/gtpolyp'
    outputdir = '/Users/jiaxin/MICCAI2020/CVC-EndoSceneStill/test-300'

    for index, file in enumerate(os.listdir(imgdir)):
        imgname = os.path.join(imgdir, file)
        image = cv2.imread(imgname)
        # 维度HWC(HW对应像素坐标的y,x)

        # cv2show(image, 'a')

        # 轮廓 最小外接矩形
        contours, hierarchy = contour_detect(image)
        positions = pts_extract(contours)

        # 将最小外接矩形写入到pascal voc格式的xml文件
        pascal_xml_writer(file, image, positions, outputdir)

        # 在mask涂上叠加矩形框
        # drawbox(positions, image)

        # 产生只有矩形框的mask并保存（可选同时画轮廓叠加验证）
        # new_image = np.zeros(image.shape, dtype=np.uint8)
        # rectangle_mask(positions, new_image)
        # cv2.drawContours(image, contours, -1, (0, 0, 255), 3)
        # cv2.imwrite(os.path.join(outputdir, file), new_image)

