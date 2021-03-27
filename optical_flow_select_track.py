# Author: Klaus
# Date: 2021/3/27 11:03
import cv2 as cv
import numpy as np

if __name__ == "__main__":
    cap = cv.VideoCapture("../doc/helix.mp4")   #选择视频文件，括号内为0则实时读取摄像头
    ret, old_frame = cap.read()
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)    #第一帧，灰度图

    point_selected = False
    point = ()
    old_points = np.array([[]])
    #lukas_kanade 参数
    lk_params = dict(winSize=(10, 10),
                     maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    #鼠标点击回调函数
    def select_point(event, x, y, flags, params):
        global point, point_selected, old_points
        if event == cv.EVENT_LBUTTONDOWN:
            point = (x,y)
            point_selected = True
            old_points = np.array([[x, y]], dtype=np.float32)   #将选中的点转化为数组格式
            cv.circle(old_frame, point, 5, (241, 164, 126), 2)
            cv.imshow('frame', old_frame)


    cv.namedWindow("frame", cv.WINDOW_AUTOSIZE)
    cv.setMouseCallback("frame", select_point)
    cv.imshow('frame', old_frame)  # 显示图片
    cv.waitKey(0)  # 按下任意键退出窗口
    cv.destroyAllWindows()

    mask = np.zeros_like(old_frame)

    while(1):
        ret, frame = cap.read()
        if ret == True:
            # cv.imshow(",", frame)
            # cv.waitKey(0)
            new_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)    #将每一帧转化为灰度图
            if point_selected == True:
                cv.circle(frame, point, 5, (241, 164, 126), 2)    #在每一帧上的目标处画小圈
                new_points, status, error = cv.calcOpticalFlowPyrLK(old_gray, new_gray, old_points, None, **lk_params)  #算法解算

                x0, y0 =old_points.ravel()
                x1, y1 = new_points.ravel()
                frame = cv.circle(frame, (int(x1),int(y1)), 5, (241, 164, 126),-1)
                mask = cv.line(mask, (int(x0), int(y0)), (int(x1), int(y1)), (0,0,255), 2)  #画运动路径
                img = cv.add(frame, mask)
                cv.imshow('frame', img)
                old_gray = new_gray.copy()  #新旧循环
                old_points = new_points
                k = cv.waitKey(24) & 0xff   #按esc退出
                if k == 27:
                    break
        else:
            break

    cap.release()
    cv.destroyAllWindows()



