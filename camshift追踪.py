# Author: Klaus
# Date: 2021/3/25 20:51
import numpy as np
import cv2 as cv

if __name__ == "__main__":
    cap = cv.VideoCapture("../doc/海星漂动—裁剪.mp4")#读取视频文件
    ret, frame=cap.read() #获取第一帧图像
    #获取图像的属性（宽和高，）,并将其转换为整数
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # 创建保存视频的对象，设置编码格式，帧率，图像的宽高等
    out = cv.VideoWriter('../doc/camshift_starfish.avi', cv.VideoWriter_fourcc('D', 'I', 'V', 'X'), 24, (frame_width, frame_height))
    #获取第一帧roi坐标
    if ret==True:
        # 像素点坐标初定义
        pro_x = []
        pro_y = []
        # 定义鼠标点击事件并将点击坐标输入数组
        def mouse_img_cod(event, cod_x, cod_y, flags, param):
            if event == cv.EVENT_LBUTTONDOWN:
                xy = "%d,%d" % (cod_x, cod_y)
                cv.circle(frame, (cod_x, cod_y), 1, (255, 0, 0), thickness=-1)
                cv.putText(frame, xy, (cod_x, cod_y), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=1)  # 将坐标值放在图片内
                cv.imshow("image", frame)
                pro_x.append(cod_x)
                pro_y.append(cod_y)

        cv.namedWindow('image', cv.WINDOW_AUTOSIZE)  # 创建一个名为image的窗口
        cv.setMouseCallback("image", mouse_img_cod)  # 鼠标事件回调
        cv.imshow('image', frame)  # 显示图片
        cv.waitKey(0)  # 按下任意键退出窗口
        cv.destroyAllWindows()

        print(pro_x[0], pro_y[0], pro_x[1], pro_y[1] )  # 打印坐标值

    track_window=(pro_x[0], pro_y[0], pro_x[1]-pro_x[0], pro_y[1]-pro_y[0])
    roi = frame[pro_y[0]:pro_y[1], pro_x[0]:pro_x[1]] #指定目标的感兴趣区域
    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV) # 转换色彩空间（HSV）
    # mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.))) #去除低亮度的值
    roi_hist = cv.calcHist([hsv_roi], [0], None, [180], [0, 180]) #计算直方图
    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX) # 归一化

    # 目标追踪
    term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1) #设置窗口搜索终止条件：最大迭代次数，窗口中心漂移最小值
    while (True):
        # 4.2 获取每一帧图像
        ret, frame = cap.read()
        if ret == True:
            # 4.3 计算直方图的反向投影
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

            # 4.4 进行meanshift追踪
            ret, track_window = cv.CamShift(dst, track_window, term_crit)

            # 4.5 将追踪的位置绘制在视频上，并进行显示
            pts = cv.boxPoints(ret)
            pts = np.int0(pts)
            img2 = cv.polylines(frame, [pts], True, 255, 2)
            cv.imshow('img2', img2)
            # 将每一帧图像写入到输出文件中
            out.write(img2)

            if cv.waitKey(30) & 0xFF == ord('q'):
                break
        else:
            break
    # 5. 资源释放
    cap.release()
    cv.destroyAllWindows()