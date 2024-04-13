import os
import time

from ppadb.client import Client as AdbClient
import random
import win32gui
import win32con
import win32ui
import pyautogui
from PIL import Image
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


def background_screenshot(hwnd, width, height):
    wDC = win32gui.GetWindowDC(hwnd)
    dcObj = win32ui.CreateDCFromHandle(wDC)
    cDC = dcObj.CreateCompatibleDC()
    dataBitMap = win32ui.CreateBitmap()
    dataBitMap.CreateCompatibleBitmap(dcObj, width, height)
    cDC.SelectObject(dataBitMap)
    cDC.BitBlt((0, 0), (width, height), dcObj, (0, 0), win32con.SRCCOPY)
    bmpinfo = dataBitMap.GetInfo()
    bmpstr = dataBitMap.GetBitmapBits(True)
    im = Image.frombuffer('RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
                          bmpstr, 'raw', 'BGRX', 0, 1)
    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())
    return im

def screenshot():
    pipe = subprocess.Popen("adb shell screencap -p",
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, shell=True)
    image_bytes = pipe.stdout.read().replace(b'\r\n', b'\n')
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    return image


def auto_resortie():
    while True:
        img = screenshot()

        img = preprocessing(img)

        height, width = img.shape[:2]
        half_width = width // 2
        half_height = height // 2

        threshold = 0.84  # 0~1의 값. 높으면 적지만 정확한 결과. 낮으면 많지만 낮은 정확도.

        error_check(img, threshold)

        img = img[half_height:, half_width:]

        restart = cv2.imread('restart.png')
        restart = preprocessing(restart)

        w, h = restart.shape[::-1]  # 타겟의 크기값을 변수에 할당

        res = cv2.matchTemplate(img, restart, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)  # res에서 threshold보다 큰 값만 취한다.

        if len(loc[0]) > 0:
            pt = (loc[1][0], loc[0][0])
            cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 0), 2)  # 결과값에 사각형을 그린다
            x_center = (2 * pt[0] + w) // 2
            y_center = (2 * pt[1] + h) // 2
            dx = w // 2
            dy = h // 2
            click([(half_width + x_center), (half_height + y_center), dx, dy])

        time.sleep(5)


def error_check(img, threshold):
    height, width = img.shape[:2]
    half_width = width // 2
    half_height = height // 2

    img = img[half_height:, half_width:]

    target = cv2.imread('full_dockyard.png')
    w,h = target.shape[::-1]
    target = preprocessing(target)

    full_res = cv2.matchTemplate(img, target, cv2.TM_CCOEFF_NORMED)
    full_loc = np.where(full_res > threshold)
    if len(full_loc[0]) > 0:
        pt = (full_loc[1][0], full_loc[0][0])
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 0), 2)  # 결과값에 사각형을 그린다
        cv2.imshow('test', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def init():
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
