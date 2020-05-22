# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os.path
from PIL import Image as image
# import PIL.Image as image
from sklearn.cluster import KMeans
import glob

import ctypes
import os

#add a border to the image for cutting
def Black_box(img):
    top_size, bottom_size, left_size, right_size = (10, 10, 10, 10)
    constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_CONSTANT,
                                  value=(0, 0, 0))  #
    return constant
#add a border to the image for cutting
def White_box(img):
    top_size, bottom_size, left_size, right_size = (10, 10, 10, 10)
    constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_CONSTANT,
                                  value=(255, 255, 255))  #
    return constant
#image denoising
def img_denoising(img):
    imgdenoising = cv2.medianBlur(img, 3)
    return imgdenoising

# image enhance
def img_enhance(img):
    labimg= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    channels = cv2.split(labimg)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe.apply(channels[0], channels[0])
    labimg = cv2.merge(channels)
    aheimg = cv2.cvtColor(labimg, cv2.COLOR_LAB2BGR)
    # return ghepimg
    return aheimg

#image cluster
def img_cluster(infile,outfile):
    #image data load
    f = open(infile, 'rb')
    data = []
    img = image.open(f)
    m, n = img.size
    for i in range(m):
        for j in range(n):
            x, y, z = img.getpixel((i, j))
            data.append([x / 256.0, y / 256.0, z / 256.0])
    f.close()
    imgData, row, col = np.mat(data),m,n
    #cluster start
    label = KMeans(n_clusters=3).fit_predict(imgData)
    label = label.reshape([row, col])
    pic_new = image.new("L", (row, col))
    a = 0
    b = 0
    c = 0
    a1 = 0
    b1 = 0
    c1 = 0
    for i in range(row):
        for j in range(col):
            if label[i][j] == 0:
                a = a + 1
            elif label[i][j] == 1:
                b = b + 1
            else:
                c = c + 1
    a0 = a
    b1 = b
    c2 = c
    d = 0
    if a < b:
        d = a
        a = b
        b = d
    if c > b:
        d = c
        c = b
        b = d
    if a < b:
        d = a
        a = b
        b = d
    if a0 == a:
        for i in range(row):
            for j in range(col):
                if label[i][j] == 0:
                    pic_new.putpixel((i, j), 255)
    if a0 == b:
        for i in range(row):
            for j in range(col):
                if label[i][j] == 0:
                    pic_new.putpixel((i, j), 100)
    if a0 == c:
        for i in range(row):
            for j in range(col):
                if label[i][j] == 0:
                    pic_new.putpixel((i, j), 0)
    if b1 == a:
        for i in range(row):
            for j in range(col):
                if label[i][j] == 1:
                    pic_new.putpixel((i, j), 255)
    if b1 == b:
        for i in range(row):
            for j in range(col):
                if label[i][j] == 1:
                    pic_new.putpixel((i, j), 100)
    if b1 == c:
        for i in range(row):
            for j in range(col):
                if label[i][j] == 1:
                    pic_new.putpixel((i, j), 0)
    if c2 == a:
        for i in range(row):
            for j in range(col):
                if label[i][j] == 2:
                    pic_new.putpixel((i, j), 255)
    if c2 == b:
        for i in range(row):
            for j in range(col):
                if label[i][j] == 2:
                    pic_new.putpixel((i, j), 100)
    if c2 == c:
        for i in range(row):
            for j in range(col):
                if label[i][j] == 2:
                    pic_new.putpixel((i, j), 0)
    pic_new.save(outfile, "JPEG")
#image_resize
def img_resize(infile, outfile, width=1400, height=1400):
    src = cv2.imread(infile, cv2.IMREAD_ANYCOLOR)
    try:
        dst = cv2.resize(src, (int(width), int(height)),  interpolation=cv2.INTER_CUBIC)
        # height, width, channels = dst.shape
        # print("width:%s,height:%s,channels:%s" % (width, height, channels))
        # for row in range(height):
        #     for list in range(width):
        #         for c in range(channels):
        #             pv = dst[row, list, c]
        #             dst[row, list, c] = 255 - pv
        cv2.imwrite(outfile, dst)
    except Exception as e:
        print(e)

#image segmentation
def img_segmentation(originalfile,clusterfile,outdir):
    imgname=os.path.basename(originalfile).split('.')[0]
    imgin = cv2.imread(clusterfile)
    gray = cv2.cvtColor(imgin, cv2.COLOR_RGB2GRAY)
    ret, dst = cv2.threshold(gray, 55, 255, cv2.THRESH_BINARY)
    dst1 = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # cv2.imwrite(outdir + "\\" + imgname + "_dst1.jpg", dst1)
    dst2 = cv2.morphologyEx(dst1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # cv2.imwrite(outdir + "\\" + imgname + "_dst2.jpg", dst2)
    binary,contours, hierarchy = cv2.findContours(dst2, 1, 5)
    # cv2.imwrite(outdir + "\\" + imgname + "_binary.jpg", binary)
    i = 0
    a = 0
    img2 = cv2.imread(originalfile)
    img1=img2.copy()
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    for cnt in contours:
        M = cv2.moments(cnt)
        i += 1
        area = cv2.contourArea(cnt)
        # print(area)
        # if area<100:
        #     continue
        x, y, w, h = cv2.boundingRect(cnt)
        img1 = cv2.rectangle(img1, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 230), 1)
        newimage = img2[y - 5:y + h + 5, x - 5:x + w + 5]
        cv2.imwrite(outdir + "\\" + imgname + "_"+str(i) + ".jpg", newimage)
    cv2.imwrite(outdir + "\\" + imgname + "_draw.jpg", img1)

#image standardization
def img_standardization(infile,outfile,width=100,height=100):
    # load in the top image
    try:
        top_img = image.open(infile, 'r')
    except:
        print(infile,'open fail')
        return
    top_img_w, top_img_h = top_img.size
    #if top_img width or height greater width need resize
    is_need_resize=False
    is_height=False
    if top_img_h>top_img_w:
        temp_size = top_img_h
        is_height = True
    else:
        temp_size=top_img_w
        is_height=False
    is_need_resize = temp_size>height
    if is_need_resize:
        if is_height:
            resize_height = height
            resize_width = int(top_img_w*height/top_img_h)
        else:
            resize_height =int(top_img_h * width / top_img_w)
            resize_width = width
        top_img = top_img.resize((resize_width, resize_height), image.ANTIALIAS)
        #reset w h
        top_img_w=resize_width
        top_img_h=resize_height
    # top_img.save('1.jpg')
    # load in the bottom image
    target = np.zeros((width, height), dtype=np.uint8)
    bottom_img = image.fromarray(np.uint8(target*255))
    #convert RGB
    bottom_img=bottom_img.convert('RGB')
    # bottom_img.show(bottom_img)
    # get the size or use  if it's constant
    bottom_img_w, bottom_img_h = bottom_img.size
    # offset the top image so it's placed in the middle of the bottom image
    offset = ((bottom_img_w - top_img_w) // 2, (bottom_img_h - top_img_h) // 2)
    # embed top_img on top of bottom_img
    bottom_img.paste(top_img, offset)
    output_name = outfile
    bottom_img.save(output_name)

if __name__ == '__main__':
    # image resize if not 700*700
    #img_resize(r'data/1.jpg', r'data/514 (2)_resize.jpg', 700, 700)
    img = cv2.imread(r'data/1.jpg')
    #enhance
    img_enhance = img_enhance(img)
    #denoising
    img_denoising= img_denoising(img_enhance)
    cv2.imwrite(r'data/1_enhance.jpg', img_enhance)
    cv2.imwrite(r'data/1_denoising.jpg', img_denoising)
    #image cluster
    img_cluster(r'data/1_denoising.jpg',r'data/1_cluster.jpg')
    imgcluster = cv2.imread(r'data/1_cluster.jpg')
    #image add 10px border
    Blackbox = Black_box(img)
    cv2.imwrite(r'data/1_blackbox.jpg', Blackbox)
    whitebox = White_box(imgcluster)
    cv2.imwrite(r'data/1_whitebox.jpg', whitebox)
    #image segmentation
    img_segmentation(r'data/1_blackbox.jpg',r'data/1_whitebox.jpg',r'data/cut/')
