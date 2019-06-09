import cv2


# 摄像头类
class video:

    def __init__(self, camera_this):
        # OpenCv 调用摄像头 use camera，0表示第一个摄像头，即笔记本内建摄像头，换位文件路径即识别视频
        self.cap = cv2.VideoCapture(camera_this)
        
    def is_open(self):  # 查看摄像头是否关闭
        return self.cap.isOpened()
    
    def get_key(self, key):  # 得到属性
        return self.cap.get(propId=key)
    
    def set_key(self, key, value):
        # 设置视频参数: propId - 设置的视频参数, value - 设置的参数值
        return self.cap.set(propId=key, value=value)
    
    def read(self):  # 读取图象
        flag, img_rd = self.cap.read()
        return flag, img_rd
    
    def close(self):  
        self.cap.release()  # 释放摄像头
        cv2.destroyAllWindows()  # 删除建立的窗口 delete all the windows
