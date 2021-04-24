import torch
import cv2
import os
import numpy as np
from PIL import Image
from torchvision import transforms, utils
import pointImg
import skeletonization

class nn_node():
    def __init__(self, model, raw_img):
        self.model = model
        self.raw_img = raw_img
    
    def get_node(self):
        sk = skeletonization.skeletonization(self.raw_img)
        skeleton_img = sk.get_skeleton().astype(np.uint8)*255
        pI = pointImg.pointImg(skeleton_img)
        _, all_centroids = pI.ske2point(30)
        all_centroids = pI.optimization(self.raw_img, all_centroids)
        nodes = []
        for i in range(len(all_centroids)):
            bbox_size = 80
            x_min, x_max, y_min, y_max = max(0, all_centroids[i][0] - bbox_size), \
                min(self.raw_img.shape[0], all_centroids[i][0] + bbox_size), \
                max(0, all_centroids[i][1] - bbox_size), \
                    min(self.raw_img.shape[1], all_centroids[i][1] + bbox_size)
            bbox_poly = self.raw_img[x_min:x_max, y_min:y_max].astype(np.uint8)
            cv2.imwrite("ff.png", bbox_poly)
            
            #predict by the model
            test_img = Image.open('ff.png')
            trans = transforms.Compose([transforms.Resize([32, 32]), transforms.ToTensor()])
            test_img = trans(test_img)
            test_img = torch.reshape(test_img, (1, 3, 32, 32))
            torch.no_grad()
            output = self.model(test_img)
            pred = torch.max(output, 1)[1]
            if pred == 1:
                nodes.append(all_centroids[i])
        
        #nodes = self.node_opt(nodes)
        pI.img_point_write("test_nn.png", nodes, self.raw_img)
    
    def node_opt(self, crossing):
        nodes = []
        
        while True:
            if not len(crossing):
                return np.array(nodes)
            curr_point = crossing[0]
            delete_nodes = []
            centroid = np.zeros_like(curr_point)
            for i in range(len(crossing)):
                if np.linalg.norm(curr_point - crossing[i]) < 80:
                    delete_nodes.append(i)
                    centroid += crossing[i]
            centroid = centroid // len(delete_nodes)
            nodes.append(centroid)
            crossing = np.delete(crossing, delete_nodes, axis=0)    
            
            
img_path = "data/raw_data/009.png"
img = cv2.imread(img_path)
net = torch.load("CNN_model")
nnn = nn_node(net, img)
nnn.get_node()