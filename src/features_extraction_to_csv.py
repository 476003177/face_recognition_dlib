# coding: utf-8
# 介绍：主要利用dlib实现摄像头的人脸识别
# 参考：https://github.com/coneypo/Dlib_face_recognition_from_camera
# 作者邮箱：476003177@qq.com
# 使用方法：1、运行get_face_from_camera录入人脸信息；2、运行face_reco_from_camera识别

# 从人脸图像文件中提取人脸特征存入 CSV
# return_128d_features()          获取某张图像的128D特征
# compute_the_mean()              计算128D特征均值
import public_variable as pv
import cv2, os, dlib, csv
from skimage import io
import numpy as np


# 返回单张图像的 128D 特征
def return_128d_features(path_img, detector, predictor, face_rec):
    img_rd = io.imread(path_img)
    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB)
    faces = detector(img_gray, 1)
    # 因为有可能截下来的人脸再去检测，检测不出来人脸了
    # 所以要确保是检测到人脸的人脸图像拿去算特征
    if len(faces) != 0:
        print("%-40s %-20s" % ("检测到人脸的图像 / image with faces detected:", path_img), '\n')
        shape = predictor(img_gray, faces[0])
        face_descriptor = face_rec.compute_face_descriptor(img_gray, shape)
    else:
        face_descriptor = 0
        print("no face")

    return face_descriptor


# 将文件夹中照片特征提取出来,并对全部照片取均值
def return_features_mean_personX(path_faces_personX, detector, predictor, face_rec):
    features_list_personX = []
    photos_list = os.listdir(path_faces_personX)
    if photos_list:
        for i in range(len(photos_list)):
            photo_this = path_faces_personX + "/" + photos_list[i]
            # 调用return_128d_features()得到128d特征
            print("%-40s %-20s" % ("正在读的人脸图像 / image to read:", photo_this))
            features_128d = return_128d_features(photo_this, detector, predictor, face_rec)
            #  print(features_128d)
            # 遇到没有检测出人脸的图片跳过
            if features_128d != 0:
                features_list_personX.append(features_128d)
#             if features_128d == 0:
#                 i += 1
#             else:
#                 features_list_personX.append(features_128d)
    else:
        print("文件夹内图像文件为空 / Warning: No images in " + path_faces_personX + '/', '\n')

    # 计算 128D 特征的均值
    # N x 128D -> 1 x 128D
    if features_list_personX:
        features_mean_personX = np.array(features_list_personX).mean(axis=0)
    else:
        features_mean_personX = '0'

    return features_mean_personX


def get_features():
    # Dlib 正向人脸检测器
    detector = dlib.get_frontal_face_detector()
    # Dlib 人脸预测器
    predictor = dlib.shape_predictor(pv.predictor_path)
    # Dlib 人脸识别模型
    face_rec = dlib.face_recognition_model_v1(pv.recognition_model_path)
    # 读取某人所有的人脸图像的数据
    people = os.listdir(pv.path_photos_from_camera)
    people.sort()
    with open(pv.feature_all_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for person in people:
            print("##### " + person + " #####")
            # Get the mean/average features of face/personX, it will be a list with a length of 128D
            features_mean_personX = return_features_mean_personX(pv.path_photos_from_camera + person, detector, predictor, face_rec)
            writer.writerow(features_mean_personX)
            print("特征均值 / The mean of features:", list(features_mean_personX))
            print('\n')
        print("所有录入人脸数据存入 / Save all the features of faces registered into: " + pv.feature_all_csv_path)


if __name__ == '__main__':
    get_features()
