#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 02:42:11 2021

@author: xuery
"""

import cv2
import time
import numpy as np
import os
import copy
import matplotlib.pyplot as plt
from scipy import spatial
from skimage import morphology
from sklearn.mixture import GaussianMixture

class get_crossing():
    def __init__(self, rgb_img, rgb_img_path=None):
        if not rgb_img_path:
            self.rgb_img = rgb_img
        else:
            self.rgb_img = cv2.imread(rgb_img_path)
        self.all_centroids = []
        self.stride = 10
        self.binary_threshold = [110, 170]
        self.node_opt_thres = 20
        self.bbox_size = 10
        self.mean_dist = 0
        self.rgb_img_path = rgb_img_path
        
    
    def get_binary(self, img = None):
        if img is not None:
           #gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
           backSub = cv2.createBackgroundSubtractorMOG2()
           binary_img = backSub.apply(self.rgb_img)
           _, binary_img = cv2.threshold(binary_img, 100, 255, cv2.THRESH_BINARY_INV)
           return binary_img
        else:
            if not self.rgb_img_path:
                gray_img = self.rgb_img[:,:,0]
            else:
                gray_img = self.rgb_img[:,:,2]
        cv2.imshow('gray', gray_img)
        
        #gray_img = cv2.cvtColor(self.rgb_img,cv2.COLOR_BGR2GRAY)
        #find the binary img, considering different rope color and background, we set two possible
        #thresholds
        _, binary_img_1 = cv2.threshold(gray_img, self.binary_threshold[0], 255, cv2.THRESH_BINARY)
        _, binary_img_2 = cv2.threshold(gray_img, self.binary_threshold[1], 255, cv2.THRESH_BINARY_INV)
        if binary_img_1.sum()/255 < gray_img.shape[0] * gray_img.shape[1]/2:
            binary_img = binary_img_1
        else:
            binary_img = binary_img_2
        return binary_img
        
        
    def get_skeleton(self):
        binary_img = self.get_binary()
        cv2.imshow("binary_img", binary_img)
        
        #canny_img = cv2.Canny(gray_img, 20, 100)
        #cv2.imshow("img", binary_img)
        #cv2.waitKey(0)

        #get skeleton
        binary_img[binary_img==255] = 1
        skeleton = morphology.skeletonize(binary_img)
        cv2.imshow("ske", skeleton.astype(np.uint8)*255)
        #cv2.imwrite("my_test/presentation/skeleton.png", skeleton.astype(np.uint8)*255)
        return skeleton
        
    
    def ske2point(self):
        skeleton_img = self.get_skeleton()
        img_w, img_h = skeleton_img.shape
        point_img = np.zeros_like(skeleton_img)

        for i in range(img_w//self.stride):
            for j in range(img_h//self.stride):
                small_img = skeleton_img[i*self.stride:(i+1)*self.stride, j*self.stride:(j+1)*self.stride]
                x_idx, y_idx = small_img.nonzero()
                if len(x_idx) == 0:
                    continue
                x_center, y_center = sum(x_idx) / len(x_idx) + i * self.stride,\
                    sum(y_idx) / len(x_idx) + j * self.stride
                point_img[int(x_center)][int(y_center)] = 255
                #all_centorids stores the idx of points
                self.all_centroids.append(np.array([int(x_center), int(y_center)]))
        self.all_centroids = np.array(self.all_centroids)
    
    
    def optimization(self, save_path=None):
        #for the points in all_centroid that don't belong to the rope, delete it
        noise_idx = []
        binary_img = self.get_binary()
        for i in range(len(self.all_centroids)):
            if binary_img[int(self.all_centroids[i][0])][int(self.all_centroids[i][1])] == 0:
                noise_idx.append(i)
        self.all_centroids = np.delete(self.all_centroids, noise_idx, axis=0)
        if save_path != None:
            self.img_point_write(save_path, all_centroids, binary_img)
    
    
    def visualization(self):
        plt.plot(self.all_centroids[:,0], self.all_centroids[:,1], 'bo', ms=5)
        plt.show()
        write_img = cv2.imread(self.rgb_img_path)
        for i in range(len(self.all_centroids)):
            cv2.circle(write_img, (self.all_centroids[i, 1], self.all_centroids[i, 0]), 3, (255,255,0), -1)
        #cv2.imwrite("my_test/presentation/point.png", write_img)
    
    
    def find_neigh_points(self, tree, centroid, num_points):
        dist, near_points_idx = tree.query(centroid, k=num_points)
        if num_points != 2:
            self.mean_dist = (np.mean(dist)+ self.mean_dist) / 2           
        near_points = self.all_centroids[near_points_idx]
        #this part is used in get_branch to delete some noise branch
        if num_points == 2:
            if dist[1] < self.mean_dist * 0.7:
                return False
            else:
                return True
        return near_points[1:]
        
    
    def write_branch(self, binary_img, curr_point, neigh_points, i):
        cv2.circle(binary_img, (curr_point[1], curr_point[0]), 2, (200), -1)
        for j in range(len(neigh_points)):
            cv2.line(binary_img, (curr_point[1],curr_point[0]),\
                     (neigh_points[j][1], neigh_points[j][0]), 100, 1)
            cv2.circle(binary_img, (neigh_points[j][1], neigh_points[j][0]), 3, (200), -1)          
        #cv2.imwrite(os.path.join("my_test/new_branch/"+str(i)+".png"), binary_img)
        #cv2.imwrite(os.path.join("my_test/presentation/branch_"+str(i)+".png"), binary_img)
        
        
    def get_angle(self, base_point, point1, point2):
        unit_v1 = (base_point - point1) / np.linalg.norm(base_point - point1)
        unit_v2 = (base_point - point2) / np.linalg.norm(base_point - point2)
        dot_product = np.dot(unit_v1, unit_v2)
        if -1.0 - 1e-6 < dot_product < -1.0:
            dot_product = -1.0
        if 1.0 < dot_product < 1.0 + 1e-6:
            dot_product = 1.0
        angle = np.arccos(dot_product)
        return angle
    
    
    def get_branch(self, curr_point, neigh_points):
        if len(neigh_points) <= 3:
            #print("less than 4 neighbors")
            return False
        tree = spatial.KDTree(neigh_points)
        for i in range(len(neigh_points)):
            if not self.find_neigh_points(tree, neigh_points[i], 2) or not self.find_neigh_points(tree, curr_point, 2):
                #print("noise points")
                return False
        neigh_dist = []
        for i in range(len(neigh_points)):
            neigh_dist.append(np.linalg.norm(curr_point - neigh_points[i]))
        neigh_dist = sorted(enumerate(neigh_dist),key = lambda x:x[1])
        
        next_point_1 = neigh_points[neigh_dist[0][0]]
        next_point_2 = neigh_points[neigh_dist[1][0]]
        next_point_3 = neigh_points[neigh_dist[2][0]]
        next_point_4 = neigh_points[neigh_dist[3][0]]
        
        
        
        if np.linalg.norm(next_point_1 - next_point_3) > np.linalg.norm(next_point_2 - next_point_3):
            angle_1 = self.get_angle(next_point_2, curr_point, next_point_3)
            dist_1 = np.linalg.norm(next_point_2 - next_point_3)
            if np.linalg.norm(next_point_1 - next_point_4) > np.linalg.norm(next_point_3 - next_point_4):
                angle_2 = self.get_angle(next_point_3, next_point_2, next_point_4)
                dist_2 = np.linalg.norm(next_point_3 - next_point_4)
            else:
                angle_2 = self.get_angle(next_point_1, curr_point, next_point_4)
                dist_2 = np.linalg.norm(next_point_1 - next_point_4)
        else:
            angle_1 = self.get_angle(next_point_1, curr_point, next_point_3)
            dist_1 = np.linalg.norm(next_point_1 - next_point_3)
            if np.linalg.norm(next_point_2 - next_point_4) > np.linalg.norm(next_point_3 - next_point_4):
                angle_2 = self.get_angle(next_point_3, next_point_1, next_point_4)
                dist_2 = np.linalg.norm(next_point_3 - next_point_4)
            else:
                angle_2 = self.get_angle(next_point_2, curr_point, next_point_4) 
                dist_2 = np.linalg.norm(next_point_2 - next_point_4)
        
        if 0.2 * np.pi < min(angle_1, angle_2) < 0.6 * np.pi:
                return True
        else:
            return False
        
    
    def new_get_branch(self, tree, curr_point, neigh_points, f):
        
        branches = []
        for i in range(len(neigh_points)):
            candicate_points = self.find_neigh_points(tree, neigh_points[i], 4)
            binary_img = self.get_binary()
            cv2.circle(binary_img, (neigh_points[i][1], neigh_points[i][0]), 4, (200), -1)
            for j in range(len(candicate_points)):
                #cv2.circle(binary_img, (candicate_points[j][1], candicate_points[j][0]), 1, (200), -1)
                #cv2.line(binary_img, (candicate_points[j][1],candicate_points[j][0]),(neigh_points[i][1], neigh_points[i][0]), 100, 1)
                if self.point_in_set(candicate_points[j], neigh_points):
                    continue
                cv2.circle(binary_img, (candicate_points[j][1], candicate_points[j][0]), 2, (200), -1)
                if self.get_angle(neigh_points[i], curr_point, candicate_points[j]) > 0.7 * np.pi:
                    if np.linalg.norm(curr_point - candicate_points[j] > 60):
                        branches.append(neigh_points[i])
                    else:
                        branches.append(candicate_points[j])
                    break
            #cv2.imwrite(os.path.join("my_test/newnew_branch/"+str(i) + str(f) +".png"), binary_img)
        if len(branches) >= 3:
            return branches
        else:
            return None
        

    
    def point_in_set(self, query_point, point_set):
        for i in range(len(point_set)):
            if np.linalg.norm(query_point - point_set[i]) == 0:
                return True
        return False
        
    def node_opt(self, crossing):
        nodes = []
        while True:
            if not len(crossing):
                return np.array(nodes)
            curr_point = crossing[0][0]
            delete_nodes = []
            centroid = np.zeros_like(curr_point)
            for i in range(len(crossing)):
                if np.linalg.norm(curr_point - crossing[i][0]) < self.node_opt_thres:
                    delete_nodes.append(i)
                    centroid += crossing[i][0]
            centroid = centroid // (len(delete_nodes))
            min_dist = 9999
            min_candicate = np.zeros_like(curr_point)
            for k in range(len(delete_nodes)):
                if  np.linalg.norm(crossing[delete_nodes[k]][0] - centroid) < min_dist:
                    min_candicate = crossing[delete_nodes[k]]
            nodes.append([centroid, crossing[delete_nodes[k]][1]])
            #nodes.append(centroid)
            crossing = np.delete(crossing, delete_nodes, axis=0)     
            

    #the size of neigh_points is (num_points - 1) because it will delete the query point itself
    def find_crossing(self, img_idx = 0, num_points = 5, visual = False):
        binary_img = self.get_binary()
        self.ske2point()
        tree = spatial.KDTree(self.all_centroids)
        if visual == True:
            self.visualization()
        crossing = []
        bbox_proportion = []
        self.mean_dist = np.linalg.norm(self.all_centroids[0] - self.all_centroids[1])
        
        #constrain 1: crossing usually have more non-zero pixels
        for i in range(len(self.all_centroids)):
            
            curr_point = self.all_centroids[i]
            neigh_points = self.find_neigh_points(tree, curr_point, num_points)
            x_min, x_max, y_min, y_max = max(0, curr_point[0] - self.bbox_size), \
                min(self.rgb_img.shape[0], curr_point[0] + self.bbox_size), \
                max(0, curr_point[1] - self.bbox_size), \
                    min(self.rgb_img.shape[1], curr_point[1] + self.bbox_size)
            bbox_poly = binary_img[x_min:x_max, y_min:y_max]
            bbox_area = (x_max - x_min + 1) * (y_max - y_min + 1)
            bbox_rope_area = bbox_poly.nonzero()
            #print(bbox_area)
            #print(len(bbox_rope_area[0]))
            #print(len(bbox_rope_area[0]) / bbox_area)
            #cv2.rectangle(self.binary_img, (y_min, x_min), (y_max, x_max), 100, 2)
            #cv2.imshow("ff", self.binary_img)
            #cv2.waitKey(0)
            #binary_img = copy.deepcopy(binary_copy)
            bbox_proportion.append(len(bbox_rope_area[0]) / bbox_area)
        bbox_sort = copy.deepcopy(bbox_proportion)
        bbox_sort.sort(reverse=True)
        constrain_1 = bbox_sort[int(len(bbox_sort)*0.4)]


        for i in range(len(self.all_centroids)):
            # if does not sastify constrain 1
            if bbox_proportion[i] <= constrain_1:
                continue
            curr_point = self.all_centroids[i]
            neigh_points = self.find_neigh_points(tree, curr_point, num_points)
            #self.write_branch(curr_point, neigh_points, i)
            
            
            #constrain 2: the connected line of centroid and its neighbors
            #should be on the rope
            on_rope_count = 0
            for j in range(len(neigh_points)):
                x1, y1, x2, y2 = curr_point[0], curr_point[1], neigh_points[j][0], neigh_points[j][1]
                if x1 - x2 == 0:
                    x1 += 1
                k = (y1 - y2) / (x1 - x2)
                b = (y2 * x1 - y1 * x2) / (x1 - x2)
                line_points = []
                line_points_on_rope = []
                for x_coor in range(min(x1, x2), max(x1, x2)):
                    y_coor = int(x_coor * k + b)
                    line_points.append(np.array([x_coor, y_coor]))
                    if binary_img[x_coor][y_coor] != 0:
                        line_points_on_rope.append(np.array([x_coor, y_coor]))
                for y_coor in range(min(y1, y2), max(y1, y2)):
                    x_coor = int((y_coor - b) / k)
                    line_points.append(np.array([x_coor, y_coor]))
                    if binary_img[x_coor][y_coor] != 0:
                        line_points_on_rope.append(np.array([x_coor, y_coor]))
                if len(line_points_on_rope) > 0.95 * len(line_points):
                    on_rope_count += 1                    
            if on_rope_count < len(neigh_points):
                continue
            
            #print("processing centroid: ", i)
            #constrain 3: this function only works when neigh_points equals to 4 or 5
            if not self.get_branch(curr_point, neigh_points):
                continue
            
            
            #constrain 4: branches test
            branches = self.new_get_branch(tree, curr_point, neigh_points, i)
            if branches is None:
                continue
            
            
            self.write_branch(binary_img, curr_point, neigh_points, i)
            binary_img = self.get_binary()
            crossing.append([curr_point, branches])
        crossing = self.node_opt(np.array(crossing))
        for i in range(len(crossing)):
            cv2.circle(self.rgb_img, (crossing[i][0][1], crossing[i][0][0]), 4, (255, 0, 0), -1) 
            for j in range(len(crossing[i][1])):
                cv2.circle(self.rgb_img, (crossing[i][1][j][1], crossing[i][1][j][0]), 3, (0, 255, 0), -1) 
            #cv2.circle(self.rgb_img, (crossing[i][1], crossing[i][0]), 5, (255, 0, 0), -1)
        
        #cv2.imwrite(os.path.join("my_test/cro/crossing" + str(img_idx).zfill(3) + ".png"), self.rgb_img)
        #cv2.imwrite(os.path.join("my_test/new_interaction/pre_" + str(img_idx).zfill(3) + ".png"), self.rgb_img)
        cv2.imwrite(os.path.join("my_test/presentation/pre_interact_" + str(img_idx).zfill(3) + ".png"), self.rgb_img)

        return crossing          #crossing: each element: [[crossing coord], [branches]]
    
    
    
if __name__ == "__main__":
    '''folder_path = 'data/'
    folder_dir = os.listdir(folder_path)
    folder_dir.sort()
    count = 0
    for file in folder_dir:
        print("processing: ", file)
        count += 1
        if count < 0:
            continue
        if count > 10:
            break
        rgb_img = cv2.imread(os.path.join(folder_path+file))
        gc = get_crossing(rgb_img)
        crossing = gc.find_crossing(img_idx = count, visual=False)'''
    img_path = 'my_test/presentation/orig_img.png'
    gc = get_crossing(cv2.imread(img_path))
    crossing = gc.find_crossing(img_idx=0, visual=True)
    
        
    