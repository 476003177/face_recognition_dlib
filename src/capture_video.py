import cv2


# ����ͷ��
class video:

    def __init__(self, camera_this):
        # OpenCv ��������ͷ use camera��0��ʾ��һ������ͷ�����ʼǱ��ڽ�����ͷ����λ�ļ�·����ʶ����Ƶ
        self.cap = cv2.VideoCapture(camera_this)
        
    def is_open(self):  # �鿴����ͷ�Ƿ�ر�
        return self.cap.isOpened()
    
    def get_key(self, key):  # �õ�����
        return self.cap.get(propId=key)
    
    def set_key(self, key, value):
        # ������Ƶ����: propId - ���õ���Ƶ����, value - ���õĲ���ֵ
        return self.cap.set(propId=key, value=value)
    
    def read(self):  # ��ȡͼ��
        flag, img_rd = self.cap.read()
        return flag, img_rd
    
    def close(self):  
        self.cap.release()  # �ͷ�����ͷ
        cv2.destroyAllWindows()  # ɾ�������Ĵ��� delete all the windows
