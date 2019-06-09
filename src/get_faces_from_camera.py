# coding: utf-8
# 介绍：主要利用dlib实现摄像头的人脸识别
# 参考：https://github.com/coneypo/Dlib_face_recognition_from_camera
# 作者邮箱：476003177@qq.com
# 使用方法：1、运行get_face_from_camera录入人脸信息；2、运行face_reco_from_camera识别

from features_extraction_to_csv import get_features  # 人脸特征提取方法
from capture_video import video  # 摄像头类
import public_variable as pv
import os, shutil, cv2, dlib
import numpy as np


# 新建保存人脸图像文件和数据CSV文件夹
# mkdir for saving photos and csv
def pre_work_mkdir():
    # 新建文件夹
    if os.path.isdir(pv.path_photos_from_camera):
        pass
    else:
        os.mkdir(pv.path_photos_from_camera)


# 删除之前存的人脸数据文件夹
def pre_work_del_old_face_folders(path_photos_from_camera, feature_all_csv_path):
    # 删除之前存的人脸数据文件夹
    folders_rd = os.listdir(pv.path_photos_from_camera)
    for i in range(len(folders_rd)):
        shutil.rmtree(path_photos_from_camera + folders_rd[i])
    if os.path.isfile(feature_all_csv_path):
        os.remove(feature_all_csv_path)


def get_person_cnt(path_photos_from_camera):
    person_cnt = 0
    # 如果有之前录入的人脸，则在之前 person_x 的序号按照 person_x+1 开始录入
    if os.listdir(path_photos_from_camera):
        # 获取已录入的最后一个人脸序号
        person_list = os.listdir(path_photos_from_camera)
        person_list.sort()
        person_num_latest = int(str(person_list[-1]).split("_")[-1])
        person_cnt = person_num_latest
    return person_cnt


def get_face_image_from_camera(person_cnt, detector, cap):
    # 人脸截图的计数器 the counter for screen shoot
    cnt_ss = 0
    # 之后用来控制是否保存图像的 flag / the flag to control if save
    save_flag = 1
    # 之后用来检查是否先按 'n' 再按 's' / the flag to check if press 'n' before 's'
    press_n_flag = 0
    # 摄像头的长宽
    w_cap = cap.get_key(3)
    h_cap = cap.get_key(4)
    while cap.is_open():
        _, img_rd = cap.read()  # 得到480 * 640的图象
        kk = cv2.waitKey(1)
        img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)
        # 人脸数
        faces = detector(img_gray, 0)
        # 字体 
        font = cv2.FONT_HERSHEY_COMPLEX
        # 按下 'n' 新建存储人脸的文件夹 / press 'n' to create the folders for saving faces
        if kk == ord('n'):
            person_cnt += 1
            pv.face_path = pv.path_photos_from_camera + "person_" + str(person_cnt)
            os.makedirs(pv.face_path)
            print('\n')
            print("新建的人脸文件夹 / Create folders: ", pv.face_path)
            cnt_ss = 0  # 将人脸计数器清零 / clear the cnt of faces
            press_n_flag = 1  # 已经按下 'n' / have pressed 'n'
        # 检测到人脸 / if face detected
        if len(faces) != 0:
            # 矩形框 / show the rectangle box
            for k, d in enumerate(faces):
                # 计算矩形位置和大小：(x,y), (宽度width, 高度height)
                pos_start = tuple([d.left(), d.top()])
                pos_end = tuple([d.right(), d.bottom()])
                height = (d.bottom() - d.top())
                width = (d.right() - d.left())
                hh = int(height / 2)
                ww = int(width / 2)
                # 设置颜色 / the color of rectangle of faces detected
                color_rectangle = (255, 255, 255)
                # 判断人脸矩形框是否超出 480x640
                if (d.right() + ww) > w_cap or (d.bottom() + hh > h_cap) or (d.left() - ww < 0) or (d.top() - hh < 0):
                    cv2.putText(img_rd, "超出框", (20, 300), font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                    color_rectangle = (0, 0, 255)
                    save_flag = 0
                    if kk == ord('s'):
                        print("请调整位置 ")
                else:
                    color_rectangle = (255, 255, 255)
                    save_flag = 1
    
                cv2.rectangle(img_rd,
                              tuple([d.left() - ww, d.top() - hh]),
                              tuple([d.right() + ww, d.bottom() + hh]),
                              color_rectangle, 2)
    
                # 根据人脸大小生成空的图像 / create blank image according to the size of face detected
                im_blank = np.zeros((int(height * 2), width * 2, 3), np.uint8)
    
                if save_flag:
                    # 按下 's' 保存摄像头中的人脸到本地 / press 's' to save faces into local images
                    if kk == ord('s'):
                        # 检查有没有先按'n'新建文件夹 / check if you have pressed 'n'
                        if press_n_flag:
                            cnt_ss += 1
                            for ii in range(height * 2):
                                for jj in range(width * 2):
                                    im_blank[ii][jj] = img_rd[d.top() - hh + ii][d.left() - ww + jj]
                            # cv2截图
                            cv2.imwrite(pv.face_path + "/img_face_" + str(cnt_ss) + ".jpg", im_blank)
                            print("写入本地 / Save into：", str(pv.face_path) + "/img_face_" + str(cnt_ss) + ".jpg")
                        else:
                            print("请在按 'S' 之前先按 'N' 来建文件夹 / Please press 'N' before 'S'")
    
        # 显示人脸数 / show the numbers of faces detected
        cv2.putText(img_rd, "Faces: " + str(len(faces)), (20, 100), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
    
        # 添加说明 / add some statements
        cv2.putText(img_rd, "Face Register", (20, 40), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "N: New face folder", (20, 350), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "S: Save current face", (20, 400), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    
        # 按下 'q' 键退出 / press 'q' to exit
        if kk == ord('q'):
            break
    
        # 如果需要摄像头窗口大小可调 / uncomment this line if you want the camera window is resizeable
        # cv2.namedWindow("camera", 0)
    
        cv2.imshow("camera", img_rd)


if __name__ == '__main__':
    # 预处理
    pre_work_mkdir()
#     pre_work_del_old_face_folders() #删除之前录入的人脸数据
    person_cnt = get_person_cnt(path_photos_from_camera=pv.path_photos_from_camera)  # 得到已录入人数
    # Dlib 正向人脸检测器 / frontal face detector
    detector = dlib.get_frontal_face_detector()
    # 得到摄像头
    cap = video(pv.camera_this)
    # 开始检测
    get_face_image_from_camera(person_cnt, detector, cap)
    # 特征提取，以便于识别
    get_features()
    # 关闭
    cap.close()
