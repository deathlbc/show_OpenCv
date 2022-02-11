import dlib
import numpy as np
import cv2
from matplotlib import pyplot as plt

# create VideoCapture object
cap = cv2.VideoCapture('./video/try1.mp4')

# get original size of frame // 取得原影片畫面尺寸
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# get Frames, fps, code type
fps = cap.get(cv2.CAP_PROP_FPS)  # get Frame Per Second
F_Count = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # get total frames count

# get Frames, fps, code type
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # use FOUR Character Code // 使用 XVID 編碼(MPEG-4)

# create VideoWriter Object // 輸出影片至 output.avi
out = cv2.VideoWriter('./video/test_out.mp4', fourcc, fps, (w, h))

# ------initial settings-------
ag = 0  # frame angle
fsize = 0.1  # frame size
fw = 0  # frame weight

# create kernel // 濾波用矩陣
k = np.ones((7, 7), np.uint8)

# addition for addWeighted
epic = 0

# create Dlib face detector object  # Dlib 的人臉偵測器
detector = dlib.get_frontal_face_detector()

# set the location of Region Of Interest(ROI) // 指定 ROI 座標位置
RECT = ((480, 20), (610, 230))
(left, top), (right, bottom) = RECT

# function for ROI operation
def roiarea(frame):  # 取出 ROI 子畫面
    return frame[top:bottom, left:right]

def replaceroi(frame, roi):  # 將 ROI 區域貼回到原畫面
    frame[top:bottom, left:right] = roi
    return frame


# start playing
while cap.isOpened():
    # read success or not & read frame // ret 讀取成功與否; frame為讀取出來的每一個frame
    ret, frame = cap.read()

    # break if not readable // 未讀取成功則跳出
    if not ret:
        break
    # fixed background // 創建固定背景
    gc = np.full((h, w, 3), 0, dtype='uint8')

    # show basic info of the video(cap)  //  顯示影片基本資訊
    cv2.putText(gc, f'{cap.get(cv2.CAP_PROP_POS_FRAMES)} frames, {cap.get(cv2.CAP_PROP_POS_MSEC):.0f} ms',
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .4, (123, 245, 12), 0, cv2.LINE_AA)

    # SHOW TIME !
    # 1.rotate&fade-in
    if cap.get(cv2.CAP_PROP_POS_FRAMES) < 80:
        # control angle and size // 旋轉角度和尺寸控制
        M1 = cv2.getRotationMatrix2D((w / 2, h / 2), ag, fsize)
        frame = cv2.warpAffine(frame, M1, (w, h))
        cv2.putText(gc, "1.rotate & fade-in", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1,
                    cv2.LINE_AA)
        if ag < 360:
            ag = ag + 360 / 80  # rotate
        if fsize < 1.0:
            fsize = fsize + 0.9 / 80  # increase size
        if cap.get(cv2.CAP_PROP_POS_FRAMES) < 80 / 2 and fw < 1.00:
            fw = fw + 0.01  # add frame weight
        elif cap.get(cv2.CAP_PROP_POS_FRAMES) > 80 / 2 and fw < 1.00:
            fw = fw + 0.04  # add MORE frame weight

    # 2.Split\Merge & Star
    elif 100 <= cap.get(cv2.CAP_PROP_POS_FRAMES) < 120:
        star = cv2.xfeatures2d.StarDetector_create()  # create STAR object // 初始化STAR檢測器
        kp = star.detect(frame, None)  # get feature by STAR // 使用STAR尋找特徵點
        frame = cv2.drawKeypoints(frame, kp, None, color=-1)
        cv2.putText(gc, "2.Split&Merge&Star", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 0,
                    cv2.LINE_AA)
    elif 120 <= cap.get(cv2.CAP_PROP_POS_FRAMES) < 140:
        b, g, r = cv2.split(frame)
        frame = cv2.merge([r, g, b])
        cv2.putText(gc, "2.Split&Merge&Star", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 0,
                    cv2.LINE_AA)
    elif 140 <= cap.get(cv2.CAP_PROP_POS_FRAMES) < 160:
        star = cv2.xfeatures2d.StarDetector_create()  # create STAR object // 初始化STAR檢測器
        kp = star.detect(frame, None)  # get feature by STAR // 使用STAR尋找特徵點
        frame = cv2.drawKeypoints(frame, kp, None, color=-1)
        cv2.putText(gc, "2.Split&Merge&Star", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 0,
                    cv2.LINE_AA)
    elif 160 <= cap.get(cv2.CAP_PROP_POS_FRAMES) < 180:
        b, g, r = cv2.split(frame)
        frame = cv2.merge([r, b, g])
        cv2.putText(gc, "2.Split&Merge&Star", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 0,
                    cv2.LINE_AA)
    elif 180 <= cap.get(cv2.CAP_PROP_POS_FRAMES) < 200:
        star = cv2.xfeatures2d.StarDetector_create()  # create STAR object // 初始化STAR檢測器
        kp = star.detect(frame, None)  # get feature by STAR // 使用STAR尋找特徵點
        frame = cv2.drawKeypoints(frame, kp, None, color=-1)
        cv2.putText(gc, "2.Split&Merge&Star", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 0,
                    cv2.LINE_AA)
    elif 200 <= cap.get(cv2.CAP_PROP_POS_FRAMES) < 220:
        b, g, r = cv2.split(frame)
        frame = cv2.merge([g, r, b])
        cv2.putText(gc, "2.Split&Merge&Star", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 0,
                    cv2.LINE_AA)
    elif 220 <= cap.get(cv2.CAP_PROP_POS_FRAMES) < 240:
        star = cv2.xfeatures2d.StarDetector_create()  # 初始化STAR檢測器
        kp = star.detect(frame, None)  # 使用STAR尋找特徵點
        frame = cv2.drawKeypoints(frame, kp, None, color=-1)
        cv2.putText(gc, "2.Split&Merge&Star", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 0,
                    cv2.LINE_AA)

    # 3.DLib Face Detect
    elif 240 <= cap.get(cv2.CAP_PROP_POS_FRAMES) < 320:
        face_rects, scores, idx = detector.run(frame, 0)  # detect the face // 偵測人臉

        for i, d in enumerate(face_rects):  # get output //  取出所有偵測的結果
            x1 = d.left();
            y1 = d.top();
            x2 = d.right();
            y2 = d.bottom()
            text = f'{scores[i]:.2f}, ({idx[i]:0.0f})'

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)  # 以方框標示偵測的人臉
            cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX,  # 標示分數
                        0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(gc, "3.DLib Face Detect", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 0,
                    cv2.LINE_AA)

    # 4.HSV
    elif 320 <= cap.get(cv2.CAP_PROP_POS_FRAMES) < 400:
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        cv2.putText(gc, "4.HSV", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 0, cv2.LINE_AA)

    # 5.Canny & ROI
    elif 400 <= cap.get(cv2.CAP_PROP_POS_FRAMES) < 480:
        frame = cv2.Canny(frame, 120, 250)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        roi = roiarea(frame)  # 取出子畫面
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # BGR to HSV
        frame = replaceroi(frame, roi)  # 將處理完的子畫面貼回到原本畫面中
        cv2.rectangle(frame, RECT[0], RECT[1], (0, 0, 255), 2)  # 在 ROI 範圍處畫個框
        cv2.putText(gc, "5.Canny & ROI", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 0, cv2.LINE_AA)

    # 6.Dilate
    elif 480 <= cap.get(cv2.CAP_PROP_POS_FRAMES) < 560:
        frame = cv2.dilate(frame, k, iterations=2)
        cv2.putText(gc, "6.Dilate", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 0, cv2.LINE_AA)

    # 7.Gradient
    elif 560 <= cap.get(cv2.CAP_PROP_POS_FRAMES) < 640:
        frame = cv2.morphologyEx(frame, cv2.MORPH_GRADIENT, k)
        cv2.putText(gc, "7.Gradient", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 0, cv2.LINE_AA)

    # 8.Erode
    elif 640 <= cap.get(cv2.CAP_PROP_POS_FRAMES) < 720:
        frame = cv2.erode(frame, k, iterations=1)
        cv2.putText(gc, "8.Erode", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 0, cv2.LINE_AA)

    # 9.DoG
    elif 720 <= cap.get(cv2.CAP_PROP_POS_FRAMES) < 800:
        img_G0 = cv2.GaussianBlur(frame, (3, 3), 0)
        img_G1 = cv2.GaussianBlur(frame, (5, 5), 0)
        frame = img_G0 - img_G1
        cv2.putText(gc, "9.DoG", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 0, cv2.LINE_AA)

    # 10.Fadeout
    elif 800 <= cap.get(cv2.CAP_PROP_POS_FRAMES) < 833:
        cv2.putText(gc, "10.fade-out", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 0, cv2.LINE_AA)
        epic = epic + 10

    # -------- needed --------------------
    c = cv2.waitKey(25)  # wait 25 ms

    if c == 27:  # key escape
        break

    result = cv2.addWeighted(gc, 1, frame, fw, epic)
    cv2.imshow('result', result)
    out.write(result)  # 寫入影格
# -------- code end --------------------
cap.release()
out.release()  # 輸出用
cv2.waitKey(0)
cv2.destroyAllWindows()