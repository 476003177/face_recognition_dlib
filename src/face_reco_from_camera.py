# coding: utf-8

# 摄像头实时人脸识别
from capture_video import video  # 摄像头类
import public_variable as pv
import pandas as pd  # 数据处理的库 Pandas
import numpy as np  # 数据处理的库 numpy
import dlib, cv2


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


def get_faces_known(csv_rd):
    features_known_arr = []  # 存放所有录入人脸特征的数组
    # 读取已知人脸数据
    for i in range(csv_rd.shape[0]):
        features_someone_arr = []
        for j in range(0, len(csv_rd.ix[i, :])):
            features_someone_arr.append(csv_rd.ix[i, :][j])
        features_known_arr.append(features_someone_arr)
    print("Faces in Database：", len(features_known_arr))
    return features_known_arr


def detect_reco(features_known_arr, detector, predictor, facerec):
    while cap.is_open():
        _, img_rd = cap.read()  # 从摄像头读取图象
        kk = cv2.waitKey(1)
        # 取灰度
        img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)
        # 人脸数 faces
        faces = detector(img_gray, 0)
        # 待会要写的字体 font to write later
        font = cv2.FONT_HERSHEY_COMPLEX
        # 存储当前摄像头中捕获到的所有人脸的坐标/名字
        pos_namelist, name_namelist = [], []
        # 按下 q 键退出
        if kk == ord('q'):
            break
        else:
            # 检测到人脸 when face detected
            if len(faces) != 0:
                # 获取当前捕获到的图像的所有人脸的特征，存储到 features_cap_arr
                features_cap_arr = []
                for i in range(len(faces)):
                    shape = predictor(img_rd, faces[i])
                    features_cap_arr.append(facerec.compute_face_descriptor(img_rd, shape))
    
                # 遍历捕获到的图像中所有的人脸
                for k in range(len(faces)):
                    # 让人名跟随在矩形框的下方
                    # 确定人名的位置坐标
                    # 先默认所有人不认识，是 unknown
                    name_namelist.append("unknown")
    
                    # 每个捕获人脸的名字坐标 the positions of faces captured
                    pos_namelist.append(tuple([faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
    
                    # 对于某张人脸，遍历所有存储的人脸特征
                    for i in range(len(features_known_arr)):
                        # 如果 person_X 数据不为空
                        if str(features_known_arr[i][0]) != '0.0':
                            print("with person_", str(i + 1), "the ", end='')
                            compare = return_euclidean_distance(features_cap_arr[k], features_known_arr[i])
                            if compare == "same":  # 找到了相似脸
                                # 在这里修改 person_1, person_2 ... 的名字
                                # 这里只写了前三个
                                # 可以在这里改称 Jack, Tom and others
                                name_namelist[k] = "Person " + str(i + 1) 
                    # 矩形框
                    for kk, d in enumerate(faces):
                        # 绘制矩形框
                        cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (0, 255, 255), 2)
    
                # 在人脸框下面写人脸名字
                for i in range(len(faces)):
                    cv2.putText(img_rd, name_namelist[i], pos_namelist[i], font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
    
        print("Faces in camera now:", name_namelist, "\n")
    
        cv2.putText(img_rd, "Press 'q': Quit", (20, 450), font, 0.8, (84, 255, 159), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Face Recognition", (20, 40), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Faces: " + str(len(faces)), (20, 100), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
    
        # 窗口显示
        cv2.imshow("camera", img_rd)


if __name__ == '__main__':
    # 处理存放所有人脸特征的 csv
    csv_rd = pd.read_csv(pv.feature_all_csv_path, header=None)
    # 读取所有已经录入的人脸特征
    features_known_arr = get_faces_known(csv_rd)
    
    # Dlib 检测器和预测器
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(pv.predictor_path)
    # 人脸识别模型，提取128D的特征矢量
    # Refer this tutorial: http://dlib.net/python/index.html#dlib.face_recognition_model_v1
    facerec = dlib.face_recognition_model_v1(pv.recognition_model_path)
    
    # 获取摄像头并设置参数
    cap = video(pv.camera_this)
    cap.set_key(3, 480)
    
    # 正式开始检测识别
    detect_reco(features_known_arr, detector, predictor, facerec)
    cap.close()
