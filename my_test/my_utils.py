#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 23:50:08 2021

@author: xuery
"""
import os.path as osp
import argparse
import numpy as np
import cv2
import pyflex
import copy
from PIL import Image
import pickle
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_gif
from my_test import get_crossing
from softgym.action_space.action_space import PickerQPG
from softgym.action_space.action_space import PickerPickPlace
from softgym.action_space.action_space import Picker
from softgym.action_space.action_space import ActionToolBase
import softgym.action_space as sa


'''
this function determine if the neigh points of the crossing is "top" (labeled as red) or "bottom" (labeled as green)
param PQ: initilized class in action_space
param PIC: initilized class in action_space
param crossing_pair: [crossing, branches], branches: [[x1, y1], [x2, y2], ...], here [x1, y1] is the neigh point of this crossing
param crossing_idx: used for saving the image
param prev_picker_pos, pre_particle_pos: after interaction of each branch, reset the rope to the original config
'''

def relative_location(env, PQ, PIC, crossing_pair, img_size, crossing_idx, prev_picker_pos, prev_particle_pos):
    top_branches = []
    bottom_branches = []
    
    sa.action_space.Picker._set_pos(prev_picker_pos, prev_particle_pos)
    cro_world_coor = PQ._get_world_coor_from_image(crossing_pair[0][1], crossing_pair[0][0])
    cro_world_coor[1] = 0.0245
    #print("crossing coordinate in 2d image: ", crossing_pair[0])
    #print("crossing coordinate in 3d space: ", cro_world_coor)
    write_img = env.get_image()
    write_img = cv2.cvtColor(write_img, cv2.COLOR_RGB2BGR)
    
    #loop for each branch of this crossing
    for i in range(len(crossing_pair[1])):
        hold_pos_2d = crossing_pair[1][i]
        #print("hold point in 2d image: ", hold_pos_2d)
        hold_world_coor = PQ._get_world_coor_from_image(hold_pos_2d[1], hold_pos_2d[0])
        hold_world_coor[1] = 0
        #print("hold point in 3d space: ", hold_world_coor)
        interaction_picker_pos = np.vstack([cro_world_coor, hold_world_coor])
        sa.action_space.Picker._set_pos(interaction_picker_pos, prev_particle_pos)
        action = np.array([[0,0.4,0,1],[0,0,0,1]])             #pick up the crossing, hold the branch
        
        
        #visualization of picked particle and held particle
        obs, _, _, info = env.step(action, record_continuous_video=True, img_size=img_size, hold=True)
        particle_x = prev_particle_pos[:, 0]
        particle_z = prev_particle_pos[:, 1]
        particle_y = prev_particle_pos[:, 2]
        fig=plt.figure()
        ax2 = Axes3D(fig)
        ax2.scatter3D(particle_x,particle_y,particle_z, cmap='Blues')
        ax2.plot3D(particle_x,particle_y,particle_z,'gray')
        #plt.show()
        
        #visualization of rope status after interaction
        _, visual_particle_pos = sa.action_space.Picker._get_pos()
        particle_x = visual_particle_pos[:, 0]
        particle_z = visual_particle_pos[:, 1]
        particle_y = visual_particle_pos[:, 2]
        fig=plt.figure()
        ax2 = Axes3D(fig)
        ax2.scatter3D(particle_x,particle_y,particle_z, cmap='Blues')
        ax2.plot3D(particle_x,particle_y,particle_z,'gray')
        #plt.show()
        
        curr_picker_pos, _ = sa.action_space.Picker._get_pos()
        
        #if the distance of any two neighbor particles is exceed the @param thers, this branch is labeled as "top", o.w., "bottom"
        if PIC.check_action_valid(thres=0.1):
            cv2.circle(write_img, (crossing_pair[1][i][1], crossing_pair[1][i][0]), 3, (0, 255, 0), -1)
            bottom_branches.append(hold_pos_2d)
        else:
            cv2.circle(write_img, (crossing_pair[1][i][1], crossing_pair[1][i][0]), 3, (0, 0, 255), -1)
            top_branches.append(hold_pos_2d)
        env.reset()
    
    
    sa.action_space.Picker._set_pos(curr_picker_pos, prev_particle_pos)       #set the rope config to its original config (before interaction)
    cv2.circle(write_img, (crossing_pair[0][1], crossing_pair[0][0]), 3, (255, 0, 0), -1)     #crossing is labeled as blue
    #cv2.imwrite(osp.join('my_test/interaction/interaction-' + str(crossing_idx) + '.png'), write_img)
    cv2.imwrite(osp.join('my_test/presentation/interaction-' + str(crossing_idx) + '.png'), write_img)
    action = np.array([[2,2,2,0],[2,2,2,0]])
    _, _, _, info = env.step(action, record_continuous_video=True, img_size=img_size)    #move the pickers out of obs, this is helpful for the next loop
    return top_branches, bottom_branches




def opt_crossings(interact_cros, gc):
    scored_cros =  []
    for i in range(len(interact_cros)):
        curr_cros = interact_cros[i][0]
        curr_top_branches = interact_cros[i][1][0]
        curr_bottom_branches = interact_cros[i][1][1]
        
        if not curr_top_branches or not curr_bottom_branches:
            continue
        if len(curr_top_branches) - len(curr_bottom_branches) == 2 or len(curr_top_branches) - len(curr_bottom_branches) == -2:
            continue
        if len(curr_top_branches) == 1 and len(curr_bottom_branches) == 1:
            continue 
        
        if len(curr_top_branches) == 2 and len(curr_bottom_branches) == 2:
            top_branch_angle = gc.get_angle(curr_cros, curr_top_branches[0], curr_top_branches[1])
            bottom_branch_angle = gc.get_angle(curr_cros, curr_bottom_branches[0], curr_bottom_branches[1])
            if top_branch_angle > 0.8 * np.pi and bottom_branch_angle > 0.8 * np.pi:
                score = 1
            elif top_branch_angle > 0.8 * np.pi or bottom_branch_angle > 0.8 * np.pi:
                score = 0.9
            elif top_branch_angle > 0.7 * np.pi or bottom_branch_angle > 0.7 * np.pi:
                score = 0.8
            else:
                continue
        
        if len(curr_top_branches) == 2 and len(curr_bottom_branches) == 1:
            score = 0.8
        if len(curr_top_branches) == 1 and len(curr_bottom_branches) == 2:
            score = 0.7    
        scored_cros.append([interact_cros[i], score])
        '''
        interact_cros[i]: [crossing_coord, (branches)]
        (branches): (top_branches, bottom_branches)
        '''
    
    delete_idx = []
    new_cros = []
    counted_idx = []
    if scored_cros:
        for i in range(len(scored_cros)):
            new_crossing = scored_cros[i][0][0]
            for j in range(i, len(scored_cros)):
                if j in counted_idx:
                    continue
                if i != j:
                    if np.linalg.norm(scored_cros[i][0][0] - scored_cros[j][0][0]) < 25:
                        if scored_cros[i][1] < scored_cros[j][1]:
                            delete_idx.append(i)
                        elif scored_cros[i][1] == scored_cros[j][1]:
                            new_cro = [[(new_crossing + scored_cros[j][0][0])//2, scored_cros[i][0][1]], scored_cros[i][1]]
                            new_cros.append(new_cro)
                            delete_idx.append(i)
                            delete_idx.append(j)
                            counted_idx.append(j)
                        elif scored_cros[i][1] > scored_cros[j][1]:
                            delete_idx.append(j)
    scored_cros = np.delete(scored_cros, delete_idx, axis=0)
    scored_cros = scored_cros.tolist()
    if new_cros:
        if scored_cros:
            for c in range(len(new_cros)):
                scored_cros.append(new_cros[c])
        else:
            scored_cros = new_cros
                
        
    return scored_cros














