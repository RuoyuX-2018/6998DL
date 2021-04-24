#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 15:47:45 2021

@author: xuery
"""

import cv2
import time
import numpy as np
import os
import copy
import pickle
import random
import math
import matplotlib.pyplot as plt
from scipy import spatial
from skimage import morphology
from sklearn.mixture import GaussianMixture
from shapely.geometry import LineString, Point
from mpl_toolkits.mplot3d import Axes3D


class get_graph():
    def __init__(self, raw_img):
        self.raw_img = cv2.resize(raw_img, (512,512))
        self.stride = 30
        self.all_centroids = []
        
    def get_binary(self):
        gray_img = cv2.cvtColor(self.raw_img, cv2.COLOR_RGB2GRAY)
        _, binary_img = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY_INV)
        return binary_img
    
    
    
    def ske2point(self):
        skeleton_img = self.get_binary()
        img_w, img_h = skeleton_img.shape

        for i in range(img_w//self.stride):
            for j in range(img_h//self.stride):
                small_img = skeleton_img[i*self.stride:(i+1)*self.stride, j*self.stride:(j+1)*self.stride]
                x_idx, y_idx = small_img.nonzero()
                if len(x_idx) == 0:
                    continue
                x_center, y_center = sum(x_idx) / len(x_idx) + i * self.stride,\
                    sum(y_idx) / len(x_idx) + j * self.stride
                #all_centorids stores the idx of points
                self.all_centroids.append(np.array([int(x_center), int(y_center)]))
        self.all_centroids = np.array(self.all_centroids)
        self.centroids_copy = copy.deepcopy(self.all_centroids)
        
        
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
        self.optimization()
        plt.plot(self.all_centroids[:,0], self.all_centroids[:,1], 'bo', ms=5)
        plt.show()


    def graph(self, num_neigh_points = 10):
        self.ske2point()
        self.visualization()
        tree = spatial.KDTree(self.all_centroids)
        start_point = [500, 0]
        neigh_points_idx, neigh_points = self.find_neigh_points(tree, start_point, 2)
        next_point = neigh_points[0]
        query_pair = [start_point, next_point]
        point_order = query_pair
        while True:
            if len(self.all_centroids) < num_neigh_points:
                break
            if len(self.all_centroids) == 30:
                break
            tree = spatial.KDTree(self.all_centroids)
            neigh_points_idx, neigh_points = self.find_neigh_points(tree, query_pair[1], num_neigh_points)
            idx, next_point = self.find_path(query_pair, neigh_points)
            if idx == -99:
                print("end of construction...")
                return point_order
            query_pair = [query_pair[1], next_point]
            point_order.append(next_point)
            #pop out the walked point
            self.all_centroids = self.all_centroids.tolist()
            self.all_centroids.pop(neigh_points_idx[idx])
            self.all_centroids = np.array(self.all_centroids)
            print("remain lens of points: ", len(self.all_centroids))
        return point_order
    
    
    def find_neigh_points(self, tree, centroid, num_points):
        dist, near_points_idx = tree.query(centroid, k=num_points)   
        near_points = self.all_centroids[near_points_idx]
        return near_points_idx[1:], near_points[1:]
    
    
    def find_path(self, query_pair, neigh_points):
        v_query = query_pair[1] - query_pair[0]
        next_point = np.zeros_like(query_pair[0])
        angle_diff = np.pi
        next_idx = -99
        for i in range(len(neigh_points)):
            v_compare =  query_pair[1] - neigh_points[i]
            #if the dist of all neigh_points is more than 65, break. This setting is for noise
            if np.linalg.norm(v_compare) >70:
                continue
            #calculate the angle of two vectors
            unit_v1 = v_query / np.linalg.norm(v_query)
            unit_v2 = v_compare / np.linalg.norm(v_compare)
            dot_product = np.dot(unit_v1, unit_v2)
            angle = np.arccos(dot_product)     #radian
            if np.pi - angle < angle_diff:
                next_point = neigh_points[i]
                angle_diff = np.pi - angle
                next_idx = i
        return next_idx, next_point
    
    
    def find_crossing(self, point_order, visual=False):
        #create lines
        pairs = []
        crossing = []
        for i in range(len(point_order)-1):
            new_pair = np.array([point_order[i], point_order[i+1]])
            pairs.append(new_pair)
        for i in range(len(pairs)):
            for j in range(len(pairs)-i):
                intersec = self.intersection(pairs[i], pairs[j+i])
                if intersec is not False:
                    crossing.append([intersec, pairs[i][0], pairs[j+i][0]])
        if visual == True:
            self.visualization_final_graph(point_order, crossing)
        return crossing
        
        
    
    #if no intersection, return False, else return the value of intersection
    def intersection(self, pair1, pair2):
        #if two pairs has a same point, break
        if np.all(pair1[0]-pair2[0]==0) or np.all(pair1[1]-pair2[0]==0) \
            or np.all(pair1[0]-pair2[1]==0) or np.all(pair1[1]-pair2[1]==0):
            return False
            
        line1 = LineString([pair1[0], pair1[1]])
        line2 = LineString([pair2[0], pair2[1]])
        intersection_point = line1.intersection(line2)
        #no intersection
        if intersection_point.is_empty:
            return False
        else:
            return np.array([intersection_point.x, intersection_point.y])
    
        
    def visualization_final_graph(self, point_order, crossing):
        x, y = zip(*point_order)
        plt.plot(x, y, '-o', zorder=1)
        crossing = np.array(crossing)
        c_x = crossing[:,0,0]
        c_y = crossing[:,0,1]
        plt.scatter(c_x, c_y, 20, 'r', zorder=2)
        plt.show()


    def trajectory(self, env, sa, point_order, crossing, stride):
        picker_pos, particle_pos = sa.action_space.Picker._get_pos()
        print(particle_pos)
        particle_dist_2d = np.linalg.norm(particle_pos[0] - particle_pos[1])
        init_particle = particle_pos[random.randint(0,len(particle_pos))].tolist()
        particle_list = []
        particle_list.append(init_particle)
        for i in range(len(point_order)-stride):
            if i % stride != 0:
                continue
            
            curr_particle = particle_list[i//stride]
            y_o = point_order[i+stride][1] - point_order[i][1]
            x_o = point_order[i+stride][0] - point_order[i][0]
            orientation = abs(y_o / x_o)
            theta = math.atan(orientation)
            if x_o == 0:
                x_o = 0.1
            if y_o == 0:
                y_o = 0.1
            x = curr_particle[0] + math.cos(theta) * particle_dist_2d * x_o / abs(x_o)
            y = curr_particle[2] + math.sin(theta) * particle_dist_2d * y_o / abs(y_o)
            next_particle = [x, curr_particle[1], y, curr_particle[3]]
            particle_list.append(next_particle)
        
        for i in range(len(particle_list)):
            if i == 3:
                particle_list[i][1] = 0.0145
            if i == 4:
                particle_list[i][1] = 0.0245
            if i == 5:
                particle_list[i][1] = 0.0145
            if i == 9:
                particle_list[i][1] = 0.0145
            if i == 10:
                particle_list[i][1] = 0.0245
            if i == 11:
                particle_list[i][1] = 0.0145
        particle_list = np.array(particle_list)
        particle_x = particle_list[:, 0]
        particle_z = particle_list[:, 1]
        particle_y = particle_list[:, 2]
        fig=plt.figure()
        ax2 = Axes3D(fig)
        ax2.scatter3D(particle_x,particle_y,particle_z, cmap='Blues')
        ax2.plot3D(particle_x,particle_y,particle_z,'gray')
        plt.show()
        return particle_list