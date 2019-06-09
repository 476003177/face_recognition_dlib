# coding: utf-8
# 介绍：主要利用dlib实现摄像头的人脸识别
# 参考：https://github.com/coneypo/Dlib_face_recognition_from_camera
# 作者邮箱：476003177@qq.com
# 使用方法：1、运行get_face_from_camera录入人脸信息；2、运行face_reco_from_camera识别

from features_extraction_to_csv import get_features  # 人脸特征提取方法
from capture_video import video  # 摄像头类
import public_variable as pv
import os, shutil, cv2, dlib


# 新建保存人脸图像文件和数据CSV文件夹
# mkdir for saving photos and csv
def pre_work_mkdir():
    # 新建文件夹
    if os.path.isdir(pv.path_photos_from_camera):
        pass
    else:
        os.mkdir(pv.path_photos_from_camera)


def get_face_image_from_camera(detector, cap):

    def put_text(press_n=0):
        if press_n > 0: 
            # 显示人脸数 / show the numbers of faces detected
            cv2.putText(img_rd, "Faces: " + str(len(faces)), (20, 100), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(img_rd, "S: Save current face", (20, 400), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        # 添加说明 / add some statements
        cv2.putText(img_rd, "Face Register", (20, 40), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "N: New face folder", (20, 350), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    
    face_path = ""
    cnt_ss = 0  # 人脸截图的计数器
    save_flag = 1  # 之后用来控制是否保存图像的 flag
    press_n = 0  # 按下"n"的次数
    # 摄像头的长宽
    w_cap = cap.get_key(3)
    h_cap = cap.get_key(4)
    # 字体 
    font = cv2.FONT_HERSHEY_COMPLEX
    while cap.is_open():
        _, img_rd = cap.read()  # 得到480 * 640的图象
        kk = cv2.waitKey(1)
        while press_n > 0:
            _, img_rd = cap.read()  # 得到480 * 640的图象
            img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)
            faces = detector(img_gray, 0)  # 人脸数
            # 检测到人脸
            if len(faces) == 1:
                face = faces[0]
                # 在人脸对应位置画出矩形框
                # 计算矩形位置和大小：(x,y), (宽度width, 高度height)
#               pos_start, pos_end = tuple([face.left(), face.top()]), tuple([face.right(), face.bottom()])
                height, width = (face.bottom() - face.top()), (face.right() - face.left())
                hh, ww = int(height / 2), int(width / 2)
                # 设置颜色
                color_rectangle = (255, 255, 255)
                # 判断人脸矩形框是否超出 480x640
                if (face.right() + ww) > w_cap or (face.bottom() + hh > h_cap) or (face.left() - ww < 0) or (face.top() - hh < 0):
                    cv2.putText(img_rd, "Out the window", (20, 300), font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                    color_rectangle = (0, 0, 255)
                    save_flag = 0
                else:
                    color_rectangle = (255, 255, 255)
                    save_flag = 1
                # 画出矩形框
                cv2.rectangle(img_rd,
                                tuple([face.left() - ww, face.top() - hh]),
                                tuple([face.right() + ww, face.bottom() + hh]),
                                color_rectangle, 2)
                # 按下 's' 保存摄像头中的人脸到本地
                if kk == ord('s'):
                    if save_flag:
                        # cv2截人脸图
                        im_blank = img_rd[face.top() - hh: face.top() - hh + height * 2, face.left() - ww:face.left() - ww + width * 2]
                        save_path = face_path + "/" + str(cnt_ss) + ".jpg"
                        cv2.imencode('.jpg', im_blank)[1].tofile(save_path)  # 不存在中文路径问题
                        print("写入本地 ：", save_path)
                        cnt_ss += 1
                    else:  # 人脸矩形框超出范围
                        print("请调整位置 ")
            elif len(faces) >= 1:
                print("检测到多个人脸，不利于录入信息，请保证每次仅有一人录入")
            put_text(press_n)  # 添加文字说明
            cv2.imshow("camera", img_rd)  # 显示窗口
            # 检测按钮
            kk = cv2.waitKey(1)
            if kk == ord('n') or kk == ord('q'): break
        
        # 按下 'n' 新建存储人脸的文件夹
        if kk == ord('n'):
            face_name = input("输入将要录入的人名：")
            face_path = pv.path_photos_from_camera + face_name.strip()
            os.makedirs(face_path)
            print("新建的人脸文件夹 : ", face_path)
            cnt_ss = 0  # 将人脸截图计数器清零 
            press_n += 1  # 已经按下 'n'
        elif kk == ord('q'):  # 按下 'q' 键退出 / press 'q' to exit
            break
        
        put_text(press_n=0)  # 添加文字说明
        # 如果需要摄像头窗口大小可调 
        # cv2.namedWindow("camera", 0)
        cv2.imshow("camera", img_rd)  # 显示窗口


if __name__ == '__main__':
    # 预处理
    pre_work_mkdir()
    shutil.rmtree(pv.path_photos_from_camera)  # 删除之前录入的人脸数据
    # Dlib 正向人脸检测器 / frontal face detector
    detector = dlib.get_frontal_face_detector()
    # 得到摄像头
    cap = video(pv.camera_this)
    # 开始检测
    get_face_image_from_camera(detector, cap)
    # 特征提取，以便于识别
    get_features()
    # 关闭
    cap.close()
