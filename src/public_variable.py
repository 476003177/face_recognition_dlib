# coding: utf-8
# 介绍：主要利用dlib实现摄像头的人脸识别
# 参考：https://github.com/coneypo/Dlib_face_recognition_from_camera
# 作者邮箱：476003177@qq.com
# 使用方法：1、运行get_face_from_camera录入人脸信息；2、运行face_reco_from_camera识别

# 公共变量文件
import  os
workplace_path = os.getcwd()
workplace_path = workplace_path.replace("\\", "/")
workplace_path = workplace_path.split("src")[0] + "src/"
data_path = workplace_path + "data/"
path_photos_from_camera = data_path + "data_faces_from_camera/"  # 保存 faces images 的路径
data_dlib_path = data_path + "data_dlib/"
feature_all_csv_path = path_photos_from_camera + "features_all.csv"
predictor_path = data_dlib_path + "shape_predictor_68_face_landmarks.dat"  # 人脸检测器
recognition_model_path = data_dlib_path + "dlib_face_recognition_resnet_model_v1.dat"  # 人脸识别模型，该模型返回的是128维向量
camera_this = 0
