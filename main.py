import os
import time

from ppadb.client import Client as AdbClient
import random
import win32api
import numpy as np
import cv2
import subprocess

deviceport = 21503  #블루스텍에서 adb 설정창에 있던 번호

memu_width = 1280
memu_height = 720

def click(cor):
    y = cor[1]
    xx = random.randint(cor[0], cor[0] + cor[2])
    yy = random.randint(y, y + cor[3])
    # 지연 시간 단위는 ms
    cmd = ("input swipe " + str(xx) + " " + str(yy) + " " + str(xx) + " " + str(yy) + " " +
           str(random.randint(54, 178)))
    device.shell(cmd)


def preprocessing(img):
    img = cv2.GaussianBlur(img, (3, 3), 3) #가우시안 블러 적용
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # trash, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #HSV형태로 변환
    coefficients = (0.001, 0, 1.2)  # (h, s, v)
    img = cv2.transform(img, np.array(coefficients).reshape((1, 3))) #색 빼주기
    # kernel = np.ones((3, 3), np.uint8)
    # img = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)

    return img

def screenshot():
    pipe = subprocess.Popen("adb shell screencap -p",
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, shell=True)
    image_bytes = pipe.stdout.read().replace(b'\r\n', b'\n')
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    return image


def auto_resortie():
    current_count = 0
    while True:
        img = screenshot()

        img = preprocessing(img)

        height, width = img.shape[:2]
        half_width = width // 2
        half_height = height // 2

        threshold = 0.83  # 0~1의 값. 높으면 적지만 정확한 결과. 낮으면 많지만 낮은 정확도.

        error_check(img, 0.9)

        img_bottom_right = img[half_height:, half_width:]

        restart = cv2.imread('assets/restart.png')
        restart = preprocessing(restart)

        w, h = restart.shape[::-1]  # 타겟의 크기값을 변수에 할당

        res = cv2.matchTemplate(img_bottom_right, restart, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        print('restart',max_val,max_loc)

        double_result = cv2.imread('assets/double_result_check.png')
        double_result = preprocessing(double_result)

        w_2,h_2 = double_result.shape[::-1]

        res_2 = cv2.matchTemplate(img_bottom_right, double_result, cv2.TM_CCOEFF_NORMED)
        _, max_val_2, _, max_loc_2 = cv2.minMaxLoc(res_2)

        if (max_val > threshold)&(max_val_2 > threshold):
            pt = (max_loc[0], max_loc[1])
            # cv2.rectangle(img_bottom_right, pt, (pt[0] + w, pt[1] + h), (0, 0, 0), 2)  # 결과값에 사각형을 그린다
            #
            # pt2 = (max_loc_2[0], max_loc_2[1])
            # cv2.rectangle(img_bottom_right, pt2, (pt2[0] + w_2, pt2[1] + h_2), (0, 0, 0), 2)  # 결과값에 사각형을 그린다
            #
            x_center = (2 * pt[0] + w) // 2
            y_center = (2 * pt[1] + h) // 2
            dx = w // 2
            dy = h // 2

            # cv2.imshow('img',img_bottom_right)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            click([(half_width + x_center), (half_height + y_center), dx, dy])
            current_count += 1
            print('해역 반복 횟수:',current_count)

        time.sleep(5)


def error_check(img, threshold):
    height, width = img.shape[:2]
    half_width = width // 2
    half_height = height // 2

    img_top_left = img[:half_height, :half_width]
    img_top_right = img[:half_height, half_width:]
    img_bottom_left = img[:half_height, half_width:]
    img_bottom_right = img[half_height:, half_width:]

    # Dockyard
    dockyard_check = cv2.imread('assets/full_dockyard.png')
    dockyard_check = preprocessing(dockyard_check)

    max_val, max_loc = image_search(dockyard_check, img_bottom_right)

    close = cv2.imread('assets/close.png')
    close = preprocessing(close)

    max_val_2, max_loc_2 = image_search(close, img_top_right)

    print(max_val)
    if (max_val > threshold)&(max_val_2 > threshold):
        print('full_dockyard founded!')
        win32api.MessageBox(0,' 자동 해역 종료 ',"alert",48)


    # # Morale
    # morale_check = cv2.imread('assets/morale.png')
    # morale_check = preprocessing(morale_check)
    #
    # image_search(morale_check, img_bottom_right, threshold)


def image_search(target, img):
    res = cv2.matchTemplate(img, target, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return max_val, max_loc

def connect_adb_device():
    os.system('cd ADB')
    os.system(f'adb connect localhost:{deviceport}')

def init():
    connect_adb_device()
    global device
    # adb settings
    adb = AdbClient(host="127.0.0.1", port=5037)
    devices = adb.devices()
    if not devices:
        print("디바이스를 찾을 수 없습니다.")
        quit()
    device = devices[0]
    if devices is not None:
        print("Adb detected")
    else:
        print("Adb not detected")
        exit(0)


if __name__ == '__main__':
    init()

    auto_resortie()
