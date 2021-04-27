# Author: Klaus
# Date: 2021/4/2 10:11
import cv2 as cv
import numpy as np
import random, math, socket
from collections import deque
from client import startServer
if __name__ == "__main__":
    cap = cv.VideoCapture("../doc/video/运动2.mp4")
    center_points = []
    target_points = []

    re, first_frame = cap.read()
    if re == True:
        # 切割图像
        h = len(first_frame)
        w = len(first_frame[0])
        first_frame = first_frame[int(1 * h / 10):, :int(2 * w / 5)]

    #选取目标点
    def mouse(event, xo, yo, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (xo, yo)
            target_points.append([xo, yo])
            cv.circle(first_frame, (xo, yo), 3, (96, 215, 30), thickness=-1)
            cv.putText(first_frame, xy, (xo, yo), cv.FONT_HERSHEY_PLAIN,
                        1.0, (255, 255, 255), thickness=1)
            cv.imshow("first_frame", first_frame)

    cv.namedWindow("first_frame", cv.WINDOW_NORMAL)

    cv.setMouseCallback("first_frame", mouse)
    cv.imshow('first_frame', first_frame)  # 显示图片
    cv.waitKey(0)  # 按下任意键退出窗口
    cv.destroyAllWindows()
    print("目标点：")
    print(target_points)

    #开启socket
    address = ('127.0.0.1', 50000)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(address)
    s.listen(5)
    ss, addr = s.accept()
    print('got connected from', addr)

    count = 0
    while(1):
        count = count + 1
        ret, frame = cap.read()
        if ret == True:
            # 切割图像
            h = len(frame)
            w = len(frame[0])
            frame = frame[int(1 *h /10):, :int(2 * w / 5)]
            # frame = cv.flip(frame, 1)   #镜像翻转
            frame_blur = cv.GaussianBlur(frame, (55,55), 0)
            hsv = cv.cvtColor(frame_blur, cv.COLOR_BGR2HSV)
            #设置颜色区间，区间内转白色，区间外转黑色
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([0, 0, 50])
            mask = cv.inRange(hsv, lower_black, upper_black)

            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
            contours, hierarchy = cv.findContours(mask.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                biggest_contour = max(contours, key=cv.contourArea)
                # 找到目标中心
                moments = cv.moments(biggest_contour)
                centre_of_contour = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
                # 发送反馈信息
                if count%100 ==0:
                    offset_x = target_points[0][0] - centre_of_contour[0]
                    offset_z = centre_of_contour[1] - target_points[0][1]
                    # print(offset_x, offset_z)
                    startServer(ss, offset_x, offset_z)

                cv.circle(frame, centre_of_contour, 5, (0, 0, 255), -1)
                # 拟合椭圆
                ellipse = cv.fitEllipse(biggest_contour)
                cv.ellipse(frame, ellipse, (0, 255, 255), 2)
                # 保存中心点
                center_points.append(centre_of_contour)

                # Draw line from center points of contour
                for i in range(1, len(center_points)):
                    if i%1 ==0 :
                        point_distance = math.sqrt(((center_points[i - 1][0] - center_points[i][0]) ** 2) + ((center_points[i - 1][1] - center_points[i][1]) ** 2))
                        if point_distance <= 50 :
                            cv.line(frame, center_points[i - 1], center_points[i], (253, 175, 1), 4)

                cv.namedWindow("original", cv.WINDOW_NORMAL)
                cv.namedWindow("mask", cv.WINDOW_NORMAL)
                cv.imshow('original', frame)
                frame_copy = frame.copy()
                cv.imshow('mask', mask)
                k = cv.waitKey(5) & 0xff  # 按esc退出
                if k == 27:
                    break
        else:
            break

    #关闭socket
    startServer(ss, 0, 0)
    s.close()
    #平滑轨迹,拟合不出来，弃用
    # all_x = [x[0] for x in center_points]
    # all_y = [x[1] for x in center_points]
    # z = np.polyfit(all_x, all_y, 70)
    # fit = np.poly1d(z)
    # print(fit)
    # y_fit = fit(all_x)
    # for index in range(1, len(all_x)):
    #     cv.line(frame_copy, (int(all_x[index-1]), int(y_fit[index-1])), (int(all_x[index]), int(y_fit[index])), (0, 0, 255), 6)
    # cv.imshow("final", frame_copy)
    # cv.waitKey(0)
    # print(center_points)
    # cv.destroyAllWindows()
    # cap.release()