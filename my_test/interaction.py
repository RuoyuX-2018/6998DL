#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 22:23:05 2021

@author: xuery
"""

import os.path as osp
import argparse
import numpy as np
import cv2
import pyflex
import time
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
import my_utils


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
    parser.add_argument('--env_name', type=str, default='RopeFlatten')
    parser.add_argument('--headless', type=int, default=0, help='Whether to run the environment with headless rendering')
    parser.add_argument('--num_variations', type=int, default=1, help='Number of environment variations to be generated')
    parser.add_argument('--save_video_dir', type=str, default='./data/', help='Path to the saved video')
    parser.add_argument('--img_size', type=int, default=256, help='Size of the recorded videos')

    args = parser.parse_args()
    
    env_kwargs = env_arg_dict[args.env_name]

    # Generate and save the initial states for running this environment for the first time
    env_kwargs['use_cached_states'] = False
    env_kwargs['save_cached_states'] = False
    env_kwargs['num_variations'] = args.num_variations
    env_kwargs['render'] = True
    env_kwargs['headless'] = args.headless

    if not env_kwargs['use_cached_states']:
        print('Waiting to generate environment variations. May take 1 minute for each variation...')
    env = SOFTGYM_ENVS[args.env_name](**env_kwargs)
    env.reset()
    #frames = [env.get_image(args.img_size, args.img_size)]

    #generate an init state
    count = 0
    while True:
        count += 1
        #the pickers should pick/unpick a segment and move randomly
        picker_pos, particle_pos = sa.action_space.Picker._get_pos()
        
        #visualize particles        
        num_particles = len(particle_pos)
        pick_id_1 = np.random.randint(num_particles)
        picked_particle_1 = particle_pos[pick_id_1, :3]
        while True:
            pick_id_2 = np.random.randint(num_particles)
            if pick_id_2 != pick_id_1:
                break
        picked_particle_2 = particle_pos[pick_id_2, :3]
        new_picker_pos = np.vstack((picked_particle_1, picked_particle_2))
        sa.action_space.Picker._set_pos(new_picker_pos, particle_pos)
        
        action = env.action_space.sample()
        _, _, _, info = env.step(action, record_continuous_video=True, img_size=args.img_size)
        #move the pickers to the boundary to avoid picker occlusion in frames
        action = np.array([[2,2,2,0],[2,2,2,0]])
        _, _, _, info = env.step(action, record_continuous_video=True, img_size=args.img_size)
    
        #wait to be stable
        for _ in range(100):
            pyflex.step()
            curr_vel = pyflex.get_velocities()
            if np.alltrue(curr_vel < 0.01):
                break
        
        z_frame = env.get_image()
        gc = get_crossing.get_crossing(z_frame)
        crossings = gc.find_crossing(img_idx = 0, visual=False)
        if crossings is not None:
            if len(crossings) >= 2:
                break
        if count > 500:
            break
    
    #change camera config to get image from different viewpoint, later use for crossing-height calculation
    #this part is not used
    camera_para = env.get_current_config()['camera_params']['default_camera']
    z_frame = Image.fromarray(env.get_image())
    z_frame.save("data/diff_view/z.png")
        
    camera_para = env.get_current_config()['camera_params']['default_camera']
    camera_para['pos'] = np.array([-0.85, 0, 0])
    camera_para['angle'] = np.array([-90 / 180. * np.pi, 0 * np.pi, 0])
    env.update_camera('default_camera', camera_para)
    x_frame = Image.fromarray(env.get_image())
    x_frame.save("data/diff_view/x.png")
        
    camera_para['pos'] = np.array([0, 0, 0.85])
    camera_para['angle'] = np.array([0 * np.pi, 0 * np.pi, -90 / 180. * np.pi])
    env.update_camera('default_camera', camera_para)
    y_frame = Image.fromarray(env.get_image())
    y_frame.save("data/diff_view/y.png")
    
    camera_para['pos'] = np.array([0, 0.85, 0])
    camera_para['angle'] = np.array([0 * np.pi, -90 / 180. * np.pi, 0 * np.pi])
    env.update_camera('default_camera', camera_para)
    z_frame = env.get_image()
    z_frame = cv2.cvtColor(z_frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite('my_test/presentation/orig_img.png', z_frame)
    #get crossing and map to world space
    start_time = time.time()
    PIC = Picker()
    gc = get_crossing.get_crossing(z_frame, 'my_test/presentation/orig_img.png')
    crossings = gc.find_crossing(img_idx = 0, visual=False)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", crossings)
    #cv2.imshow("crossing", cv2.imread("./my_test/new_interaction/pre_000.png"))
    
    world_coors = []
    if crossings is not None:
        for i in range(len(crossings)):
            PQ = PickerQPG((z_frame.shape[0],z_frame.shape[1]), camera_para['pos'], camera_para['angle'])
            cro_world_coor = PQ._get_world_coor_from_image(crossings[i][0][1], crossings[i][0][0])
            world_coors.append([cro_world_coor[0], 0.0245, cro_world_coor[2]])
        #cv2.imshow("x_frame_binary", x_frame_binary)
        
    picker_pos, particle_pos = sa.action_space.Picker._get_pos()
    world_coors = np.array(world_coors)
    particle_x = particle_pos[:, 0]
    particle_z = particle_pos[:, 1]
    particle_y = particle_pos[:, 2]
    fig=plt.figure()
    ax2 = Axes3D(fig)
    ax2.scatter3D(particle_x,particle_y,particle_z, cmap='Blues')
    ax2.plot3D(particle_x,particle_y,particle_z,'gray')
    if world_coors != []:
        ax2.scatter3D(world_coors[:,0],world_coors[:,2],world_coors[:,1], cmap='Reds')
    #plt.show()
    
    PQ = PickerQPG((z_frame.shape[0],z_frame.shape[1]), camera_para['pos'], camera_para['angle'])    #init PickerQPG class
    prev_picker_pos, prev_particle_pos = sa.action_space.Picker._get_pos()
    interact_cros = []
    for c in range(len(crossings)):
        top_branches, bottom_branches = my_utils.relative_location(env, PQ, PIC, crossings[c], args.img_size, c, prev_picker_pos, prev_particle_pos)
        interact_cros.append([crossings[c][0], (top_branches, bottom_branches)])
    scored_cros = my_utils.opt_crossings(interact_cros, gc)
    after_interact_img = env.get_image()
    after_interact_img = cv2.cvtColor(after_interact_img, cv2.COLOR_RGB2BGR)
    for i in range(len(scored_cros)):
        cv2.circle(after_interact_img, (scored_cros[i][0][0][1], scored_cros[i][0][0][0]), 3, (255, 0, 0), -1)
        top_branches = scored_cros[i][0][1][0]
        bottom_branches = scored_cros[i][0][1][1]
        if top_branches:
            for top_j in range(len(top_branches)):
                cv2.circle(after_interact_img, (top_branches[top_j][1], top_branches[top_j][0]), 3, (0, 0, 255), -1)
        if bottom_branches:
            for bottom_j in range(len(bottom_branches)):
                cv2.circle(after_interact_img, (bottom_branches[bottom_j][1], bottom_branches[bottom_j][0]), 3, (0, 255, 0), -1)
            
    #cv2.imwrite("my_test/new_interaction/after_000.png", after_interact_img)
    cv2.imwrite('my_test/presentation/after_interact_000.png', after_interact_img)
    end_time = time.time()
    print("total perception and interaction time: ", end_time - start_time)


if __name__ == '__main__':
    main()
