import cv2
import os

class dataPre:
    def __init__(self,folder_path):
        self.folder_path = folder_path
        self.folder_dir = os.listdir(folder_path)
        
    def dataReshape(self, out_path, w, h):
        for file in self.folder_dir:
            img = cv2.imread(os.path.join(self.folder_path,file))
            new_img = cv2.resize(img,(w,h),interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(out_path,file),new_img)
    

    def data_rename(self, tag = 0):
        i = 0
        for file in self.folder_dir:
            filename=os.path.splitext(file)[0]   
            filetype=os.path.splitext(file)[1]  
            old_name = self.folder_path + file
            print(old_name)
            new_name = self.folder_path + format(str(i+tag), "0>4s") + filetype
            #new_name = self.folder_path + filename + str(i) + filetype
            os.rename(old_name,new_name)
            i += 1
            

if __name__ == "__main__":
    folder_path = "data/nn_data/p/"
    pre = dataPre(folder_path)
    #out_path = "data/nn_data/p/"
    pre.data_rename(0)
    #pre.dataReshape(out_path,64, 64)
