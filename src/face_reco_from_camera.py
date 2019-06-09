# coding: utf-8
# 介绍：主要利用dlib实现摄像头的人脸识别
# 参考：https://github.com/coneypo/Dlib_face_recognition_from_camera
# 作者邮箱：476003177@qq.com
# 使用方法：1、运行get_face_from_camera录入人脸信息；2、运行face_reco_from_camera识别

# 摄像头实时人脸识别
from capture_video import video  # 摄像头类
import public_variable as pv
from PIL import Image, ImageDraw, ImageFont
import pandas as pd  # 数据处理的库 Pandas
import numpy as np  # 数据处理的库 numpy
import dlib, cv2


# 解决图片中的中文乱码问题
def change_cv2_draw(img, strs, local, sizes, colour):
    pilimg = Image.fromarray(img)
    draw = ImageDraw.Draw(pilimg)  # 图片上打印
    font = ImageFont.truetype("SIMYOU.TTF", sizes, encoding="utf-8")
    draw.text(local, strs, colour, font=font)
    image = np.array(pilimg)
    return image


# 计算两个128D向量间的欧式距离
def return_euclidean_distance(feature_1, feature_2, d=0.5):
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    print("e_distance: ", dist)
    if dist > d:
        return "diff"
    else:
        return "same"


def get_faces_known(feature_all_csv_path):
    data = pd.read_csv(feature_all_csv_path, header=None, encoding="gbk").values
    features_known_dict = {}  # 存放所有录入人名和人脸特征的字典
    # 读取已知人脸数据
    for data_this in data:
        name_this = data_this[0]
        feature_this = np.array(data_this[1:], np.float64)
        features_known_dict[name_this] = feature_this
    print("Faces in Database：", len(features_known_dict))
    return features_known_dict


def detect_reco(features_known_dict, detector, predictor, facerec):
    while cap.is_open():
        _, img_rd = cap.read()  # 从摄像头读取图象
        kk = cv2.waitKey(1)  # 按键
        img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)  # 取灰度
        faces = detector(img_gray, 0)  # 检测到的人脸
        font = cv2.FONT_HERSHEY_COMPLEX  # 字体
        # 存储当前摄像头中捕获到的所有人脸的坐标/名字
        pos_list, name_list = [], []
        # 按下 q 键退出
        if kk == ord('q'):
            break
        else:
            # 检测到人脸
            if len(faces) != 0:
                # 遍历捕获到的图像中所有的人脸
                for face in faces:
                    # 获取人脸特征，以便之后与已录入的数据进行对比
                    shape = predictor(img_rd, face)
                    feature = facerec.compute_face_descriptor(img_rd, shape)
                    # 让人名跟随在矩形框的下方
                    name = "unknown"  # 先默认不认识，是 unknown 
                    # 每个捕获人脸的名字坐标
                    pos_list.append(tuple([face.left(), int(face.bottom() + (face.bottom() - face.top()) / 4)]))
                    # 对于某张人脸，遍历所有存储的人脸特征
                    for name_this in features_known_dict:
                        feature_this = features_known_dict[name_this]
                        compare = return_euclidean_distance(feature, feature_this)
                        if compare == "same": name = name_this  # 找到了相似脸，修改名字
                    name_list.append(name) 
                    # 矩形框
                    for kk, d in enumerate(faces):
                        # 绘制矩形框
                        cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (0, 255, 255), 2)
                # 在人脸框下面写人脸名字
                for i in range(len(faces)):
                    img_rd = change_cv2_draw(img_rd, name_list[i], pos_list[i], 25, (0, 255, 255))  # 无中文显示问题
#                     cv2.putText(img_rd, name_list[i], pos_list[i], font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)  # 有中文显示问题
    
        print("Faces in camera now:", name_list, "\n")
        cv2.putText(img_rd, "Press 'q': Quit", (20, 450), font, 0.8, (84, 255, 159), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Face Recognition", (20, 40), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Faces: " + str(len(faces)), (20, 100), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
        # 窗口显示
        cv2.imshow("camera", img_rd)


if __name__ == '__main__':
    # 读取所有已经录入的人名和人脸特征
    features_known_dict = get_faces_known(pv.feature_all_csv_path)
    # Dlib 检测器和预测器
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(pv.predictor_path)
    # 人脸识别模型，提取128D的特征矢量
    facerec = dlib.face_recognition_model_v1(pv.recognition_model_path)
    
    # 获取摄像头并设置参数
    cap = video(pv.camera_this)
    cap.set_key(3, 480)
    
    # 正式开始检测识别
    detect_reco(features_known_dict, detector, predictor, facerec)
    cap.close()
