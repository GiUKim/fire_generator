import cv2
from glob import glob
import random
import os
import numpy as np

def convert(x1, x2, y1, y2, height, width):
    dw = 1./width
    dh = 1./height
    x = (x1 + x2) / 2.0
    y = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h

def reverse_convert(x, y, w, h, height, width):
    x = x * width
    w = w * width
    y = y * height
    h = h * height
    x1 = int(x - w / 2.0)
    x2 = int(x + w / 2.0)
    y1 = int(y - h / 2.0)
    y2 = int(y + h / 2.0)
    return x1, x2, y1, y2

def coordinate_align(x, y, rows, cols):
    new_x = x - rows // 2
    new_y = y - cols // 2
    if new_x < 0:
        new_x = 0
    if new_y < 0:
        new_y = 0
    return new_x, new_y

def draw_flame(x, y, src1):
    rand_idx = random.sample(range(0, len(imgs_flame)), 2)
    add1 = cv2.imread(imgs_flame[rand_idx[0]])
    #add2 = cv2.imread(imgs_flame[rand_idx[1]])
    src2 = cv2.resize(add1, (96, 96))
    #add2 = cv2.resize(add2, (96, 96))
    #src2 = cv2.add(add1, add2)

    src2 = cv2.resize(src2, None, None, fx=cur_flame_fx, fy=cur_flame_fy)
    rows, cols, channels = src2.shape

    new_x, new_y = coordinate_align(x, y, rows, cols)

    roi = src1[new_x:rows + new_x, new_y:cols + new_y]
    gray = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)
    ret, mask_inv = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)  # 배경 흰색 그림 검정
    mask = cv2.bitwise_not(mask_inv)

    src1_bg = cv2.bitwise_and(roi, roi, mask=mask)
    src2_fg = cv2.bitwise_and(src2, src2, mask=mask_inv)

    dst = cv2.bitwise_or(src1_bg, src2_fg)

    src1[new_x:rows + new_x, new_y:cols + new_y] = dst

'''
def draw_smoke(x, y, src1, src2):
    rand_idx = random.sample(range(0, len(imgs_smoke)), 2)
    add1 = cv2.imread(imgs_smoke[rand_idx[0]])
    add2 = cv2.imread(imgs_smoke[rand_idx[1]])
    add1 = cv2.resize(add1, (96, 96))
    add2 = cv2.resize(add2, (96, 96))
    src2_smoke = cv2.add(add1, add2)

    smoke_list = []
    smoke_list.append(src2_smoke)
    smoke_90 = cv2.rotate(src2_smoke, cv2.ROTATE_90_CLOCKWISE)
    smoke_list.append(smoke_90)
    smoke_180 = cv2.rotate(src2_smoke, cv2.ROTATE_180)
    smoke_list.append(smoke_180)
    smoke_270 = cv2.rotate(src2_smoke, cv2.ROTATE_90_COUNTERCLOCKWISE)
    smoke_list.append(smoke_270)
    src2 = random.choice(smoke_list)

    src2 = cv2.resize(src2, None, None, fx=cur_smoke_fx, fy=cur_smoke_fy)
    #sub = 50 * np.ones((src2.shape[1], src2.shape[0], 3), np.uint8)

    rows, cols, channels = src2.shape
    new_x, new_y = coordinate_align(x, y, rows, cols)
    roi = src1[new_x:rows + new_x, new_y:cols + new_y]
    gray = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)
    #gray = cv2.subtract(gray, 30)
    ret, mask_inv = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_not(mask_inv)
    src1_bg = cv2.bitwise_and(roi, roi, mask=mask)
    src2_fg = cv2.bitwise_and(src2, src2, mask=mask_inv)

    dst = cv2.bitwise_or(src1_bg, src2_fg)
    src1[new_x:rows + new_x, new_y:cols + new_y] = dst
'''

drawing = False
green = (0, 255, 0)
(ix, iy) = (-1, 1)
from math import sqrt, pow
def mouse_event(event, x, y, flags, param):
    global LClick, RClick
    global ix, iy, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        LClick=True
    elif event == cv2.EVENT_MOUSEMOVE and LClick:
        draw_flame(y, x, src1)
    elif event == cv2.EVENT_LBUTTONUP:
        display_src1 = get_bbox_drawing_from_src(src1, imgs_car[cur_idx])
        LClick = False

    if event == cv2.EVENT_RBUTTONDOWN and LClick==False:
        RClick=True
        drawing = True
        (ix, iy)=  x, y
    elif event == cv2.EVENT_MOUSEMOVE and RClick:
        if drawing == True:
            #cv2.rectangle(src1, (ix, iy), (x, y), green, 3)
            display_src1 = get_bbox_drawing_from_src(src1, imgs_car[cur_idx])
            #cv2.rectangle(display_src1, (ix, iy), (x, y), green, 3)
            cv2.imshow('label', display_src1)
            #cv2.imshow('label', src1)
        #draw_smoke(y, x, src1, None)
    elif event == cv2.EVENT_RBUTTONUP:
        drawing = False
        RClick = False
        display_src1 = get_bbox_drawing_from_src(src1, imgs_car[cur_idx])
        cv2.rectangle(display_src1, (ix, iy), (x, y), green,3)
        fw = open(imgs_car[cur_idx].split('.')[0] + '.txt', 'a')
        cls = 0
        nx, ny, nw, nh = convert(ix, x, iy, y, src1.shape[0], src1.shape[1])
        newline = str(cls) + ' ' + str(round(nx, 6)) + ' ' + str(round(ny, 6)) + ' ' + str(round(nw, 6)) + ' ' + str(round(nh, 6)) + '\n'
        print(newline)
        fw.write(newline)
        fw.close()
        #print(ix, iy, x, y)
        #cv2.imshow('label', display_src1)
    display_src1 = get_bbox_drawing_from_src(src1, imgs_car[cur_idx])
    cv2.imshow('label', display_src1)
    #cv2.imshow('label', src1)

imgs_car = glob('roadcar/*.jpg')
imgs_flame = glob('flame/*.jpg')
imgs_smoke = glob('smoke/*.jpg')
print(imgs_flame)
cur_smoke_size = (72, 72)
cur_flame_size = (48, 48)
cur_flame_fx = 1.0
cur_flame_fy = 1.0
cur_smoke_fx = 1.0
cur_smoke_fy = 1.0
cur_idx = 0

def get_bbox_drawing_from_src(src, img_file_path):
    bbox = np.zeros((src.shape[0], src.shape[1], 3), dtype=np.uint8)
    try:
        fr = open(img_file_path.split('.')[0] + '.txt', 'r')
        lines = fr.readlines()
        for line in lines:
            x, y, w, h = line.split(' ')[1:]
            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)
            x1, x2, y1, y2 = reverse_convert(x, y, w, h, src.shape[0], src.shape[1])
            cv2.rectangle(bbox, (x1, y1), (x2, y2), (0, 255, 0), 3)
        fr.close()
    except:
        pass
    # cv2.imshow('bbox', bbox)
    mask = cv2.cvtColor(bbox, cv2.COLOR_BGR2GRAY)
    mask_inv = cv2.bitwise_not(mask)
    box = cv2.bitwise_and(bbox, bbox, mask=mask)
    back = cv2.bitwise_and(src, src, mask=mask_inv)
    display_src1 = cv2.add(box, back)

    return display_src1


while True:
    rand_idx = random.sample(range(0, len(imgs_flame)), 2)
    src1 = cv2.imread(imgs_car[cur_idx])
    org_img_size = src1.shape[:2]
    src1 = cv2.resize(src1, (1280, 768))
    print(src1.shape)

    add1 = cv2.imread(imgs_flame[rand_idx[0]])
    add2 = cv2.imread(imgs_flame[rand_idx[1]])
    add1 = cv2.resize(add1, (96, 96))
    add2 = cv2.resize(add2, (96, 96))
    src2_flame = cv2.add(add1, add2)

    rand_idx_smoke = random.randrange(0, len(imgs_smoke))
    #src2_smoke = cv2.imread(imgs_smoke[rand_idx_smoke], cv2.IMREAD_COLOR) 

    display_src1 = get_bbox_drawing_from_src(src1, imgs_car[cur_idx])
    cv2.imshow('label', display_src1)


    cv2.setMouseCallback('label', mouse_event, src1)
    key = cv2.waitKey()

    if key == 27:
        break
    elif key == ord('1'):
        cv2.imwrite(os.path.join('C:/fire_generator/roadcar', imgs_car[cur_idx].split('\\')[-1]), src1)
        print('save file:', os.path.join('C:/fire_generator/roadcar', imgs_car[cur_idx].split('\\')[-1]))
        cur_flame_fx += 0.2
        cur_flame_fy += 0.2
    elif key == ord('2'):
        cv2.imwrite(os.path.join('C:/fire_generator/roadcar', imgs_car[cur_idx].split('\\')[-1]), src1)
        print('save file:', os.path.join('C:/fire_generator/roadcar', imgs_car[cur_idx].split('\\')[-1]))
        cur_flame_fx -= 0.2
        cur_flame_fy -= 0.2
    elif key == ord('3'):
        cv2.imwrite(os.path.join('C:/fire_generator/roadcar', imgs_car[cur_idx].split('\\')[-1]), src1)
        print('save file:', os.path.join('C:/fire_generator/roadcar', imgs_car[cur_idx].split('\\')[-1]))
        cur_smoke_fx += 0.2
        cur_smoke_fy += 0.2
    elif key == ord('4'):
        cv2.imwrite(os.path.join('C:/fire_generator/roadcar', imgs_car[cur_idx].split('\\')[-1]), src1)
        print('save file:', os.path.join('C:/fire_generator/roadcar', imgs_car[cur_idx].split('\\')[-1]))
        cur_smoke_fx -= 0.2
        cur_smoke_fy -= 0.2
    elif key == ord('a'):
        if cur_idx > 0:
            src1 = cv2.resize(src1, (org_img_size[1], org_img_size[0]))
            cv2.imwrite(os.path.join('C:/fire_generator/roadcar', imgs_car[cur_idx].split('\\')[-1]), src1)
            print('save file:', os.path.join('C:/fire_generator/roadcar', imgs_car[cur_idx].split('\\')[-1]))
            cur_idx -= 1
    elif key == ord('d'):
        if cur_idx < len(imgs_car) - 1:
            src1 = cv2.resize(src1, (org_img_size[1], org_img_size[0]))
            cv2.imwrite(os.path.join('C:/fire_generator/roadcar', imgs_car[cur_idx].split('\\')[-1]), src1)
            print('save file:', os.path.join('C:/fire_generator/roadcar', imgs_car[cur_idx].split('\\')[-1]))
            cur_idx += 1
        elif cur_idx == len(imgs_car) - 1:
            src1 = cv2.resize(src1, (org_img_size[1], org_img_size[0]))
            cv2.imwrite(os.path.join('C:/fire_generator/roadcar', imgs_car[cur_idx].split('\\')[-1]), src1)
            print('save file:', os.path.join('C:/fire_generator/roadcar', imgs_car[cur_idx].split('\\')[-1]))
            cur_idx = 0