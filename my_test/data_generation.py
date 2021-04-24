import cv2
import os
import numpy as np
import pointImg
import skeletonization


class data_generation():
    def __init__(self, raw_img):
        self.raw_img = raw_img
    
    def get_data(self, tag):
        sk = skeletonization.skeletonization(self.raw_img)
        skeleton_img = sk.get_skeleton().astype(np.uint8)*255
        pI = pointImg.pointImg(skeleton_img)
        _, all_centroids = pI.ske2point(30)
        all_centroids = pI.optimization(self.raw_img, all_centroids)
        for i in range(len(all_centroids)):
            bbox_size =50
            x_min, x_max, y_min, y_max = max(0, all_centroids[i][0] - bbox_size), \
                min(self.raw_img.shape[0], all_centroids[i][0] + bbox_size), \
                max(0, all_centroids[i][1] - bbox_size), \
                    min(self.raw_img.shape[1], all_centroids[i][1] + bbox_size)
            bbox_poly = self.raw_img[x_min:x_max, y_min:y_max].astype(np.uint8)
            cv2.imwrite(os.path.join("data/f/"+format(str(i+tag), "0>4s")+".png"), bbox_poly)
        return len(all_centroids)
    

    def make_data(self, data_path):
        
        img_data = torchvision.datasets.ImageFolder(data_path,
                                                    transform=transforms.Compose([
                                                        transforms.Resize((32, 32)),
                                                        transforms.ToTensor()]))
        
        print(len(img_data))
        data_loader = torch.utils.data.DataLoader(img_data, batch_size=20,shuffle=True)
        


        
        

if __name__ == "__main__":
    '''folder_path = "data/raw_data/"
    folder_dir = os.listdir(folder_path)
    tag = 0
    for file in folder_dir:
        img_path = os.path.join(folder_path,file)
        raw_img = cv2.imread(img_path)
        dg = data_generation(raw_img)
        num_img = dg.get_data(tag)
        tag += num_img'''
    raw_img = None
    dg = data_generation(raw_img)
    dg.make_data("data/nn_data/")
    