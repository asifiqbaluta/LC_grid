# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 13:46:38 2020

@author: iqbal
"""

import numpy as np
# import random
# import itertools
import scipy.misc
import matplotlib.pyplot as plt
import copy
from gym import spaces
# from PIL import Image
# from skimage.transform import resize

import pdb

# WallPatern:
#  center block for size 10
# for i in range(4,7):
#     for j in range(4,7):
#         self.wall.append([i,j])


def isin_(ar1,ar2):
  sz2 = np.shape(ar2)
  for i in range(sz2[0]):
    if ar1[0]==ar2[i][0]:
      if ar1[1]==ar2[i][1]:
        return True
  return False

def isin_index(ar1,ar2):
# returns ar2 index where there is a match
  sz2 = np.shape(ar2)
  for i in range(sz2[0]):
    if ar1[0]==ar2[i][0]:
      if ar1[1]==ar2[i][1]:
        return i
  return -2

def read_and_create_grid(input_file):
    if type(input_file)!=type(''):
        print("invalid input")
        return

    file = open(input_file, 'r')
    Lines = file.readlines()
    init_line = Lines[0].split()

    out = np.zeros( (int(init_line[0]),int(init_line[1]) ) ,dtype=np.int)
    
    for i in Lines[1:]:
        temp_line = i.split()        
        if temp_line[0] == 'W':            
            for j in np.arange(int(temp_line[3]),int(temp_line[4])+1):                
                out[int(temp_line[1]):int(temp_line[2])+1,j] = 1

        elif temp_line[0] == 'O':           
            for j in np.arange(int(temp_line[3]),int(temp_line[4])+1):
                out[int(temp_line[1]):int(temp_line[2])+1,j] = int(temp_line[5])

        elif temp_line[0] == 'A':
            out[int(temp_line[1])][int(temp_line[2])] = -1
            
    return out

#from IPython.display import clear_output
#import torch
#import torchvision.transforms as T

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#resize_T = T.Compose([T.ToPILImage(),
                    #T.Resize(40, interpolation=Image.CUBIC),
                    #T.ToTensor()])
class gameOb():
    def __init__(self,coordinates,size,intensity,channel,reward,name,ID):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        # self.reward = reward
        self.name = name
        self.ID = ID
        
class gameEnv():
    def __init__(self,input_file=None,_seed_id=0,res=84):#,seed_ = np.random.randint(1000)):
        if input_file==None:
            print('No input file specified')
            return 
        else:
            env_mat = read_and_create_grid("grid.txt")
        size = env_mat.shape[0]
        self.sizeX = size+2
        self.sizeY = size+2
        self.res = self.sizeX
        # self.actions =  ['no action', 'loop closure']
        # self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.res, self.res, 3),
            dtype='uint8'
        )
        self.observation_space = spaces.Dict({
            'image': self.observation_space
        }) 
        self.last_reward_and_action=[[0,0],[0,0]] 
        self.seed_id = _seed_id
        self.seed(self.seed_id) 
        self.objects = []
        self.total_reward = 0
        self.last_location = []
        self.last_location_buffer_size = 8
        self.locations_visited = np.zeros((size,size),dtype=np.bool)
        self.LC_ = self.locations_visited.copy()
        self.hero_ = [0,0]
        self.visit_count = 0
        self.wall  = []#[[3,3],[3,4],[3,5],[3,6],[3,7],[4,3],[5,3],[6,3],[7,3],[5,7],[6,7],[7,7],[7,5],[7,6]]
        # for i in range(_wall_count):
        self.item =[]
        self.item_id = []
#        print(env_mat)
        for i in range(env_mat.shape[0]):
            for j in range(env_mat.shape[1]):
                if env_mat[i,j] > 1:
                    self.item.append([i,j])
                    self.item_id.append(env_mat[i,j])
                elif env_mat[i,j] == -1:
                    self.hero_ = [i,j]
                elif env_mat[i,j] == 1:
                    self.wall.append([i,j])
        

        self.temp_x= [2,2,7,7]#,4,5,4,5]#,1,1,2]
        self.temp_y= [2,7,2,7]#,4,4,5,5]#,1,2,1]
        self.LC_loc= []
        
        # self.seed(self.seed_id)    
        self.counter = np.random.randint(2,6)

        self.actions=[0]
        for loop in range(len(self.temp_x)):
            self.LC_[self.temp_x[loop],self.temp_y[loop]]=True
            self.LC_loc.append([self.temp_x[loop],self.temp_y[loop]])
            self.actions.append(loop+1)
        i=0

        self.action_space = spaces.Discrete(len(self.actions))
        
        return
    
    def seed(self, seed=0):
        # set random seed
        self.seed_id = seed
        np.random.seed(seed)
        return
    def reset(self):
        # coordinates,size,intensity,channel,reward,name
        #self.objects = []
        self.locations_visited = np.zeros((self.sizeX-2,self.sizeY-2),dtype=np.bool)
        self.last_location = []
        # self.LC_ = self.locations_visited.copy()
        # self.locations_visited_period = np.zeros((self.sizeX-2,self.sizeY-2),dtype=np.bool)
        # self.last_locations =  np.zeros((3,2),dtype=np.uint16)
        # self.seed(self.seed_id)
#        newPoistion = [np.random.randint(1,self.sizeX-1),np.random.randint(1,self.sizeY-1)]
#        while ([newPoistion[0]-1,newPoistion[1]-1],self.wall)==True:
#            newPoistion = [np.random.randint(1,self.sizeX-1),np.random.randint(1,self.sizeY-1)]
# 
#        # hero = gameOb(newPoistion,1,10,0,None,'hero',-1)
#        self.objects[0].x = copy.deepcopy(newPoistion[0])
#        self.objects[0].y = copy.deepcopy(newPoistion[1])
        self.state = self.get_screen()
        return self.state
    def update_last_location(self,x,y):
        # self.last_locations[2,:] = self.last_locations[1,:].copy()
        # self.last_locations[1,:] = self.last_locations[0,:].copy()
        # self.last_locations[0,0] = x
        # self.last_locations[0,1] = y 
        if len(self.last_location)>self.last_location_buffer_size:
            self.last_location.pop(0)
        self.last_location.append([x,y])
    def moveChar(self,direction):
        # 0 - up, 1 - down, 2 - left, 3 - right
        heroX = self.hero_[0]#self.objects[0].x
        heroY = self.hero_[1]#self.objects[0].y
        if len(self.last_location)>self.last_location_buffer_size:
            self.locations_visited[self.last_location[0][0],self.last_location[0][1]] = True
        self.update_last_location(heroX-1,heroY-1)
        # self.locations_visited[heroX-1,heroY-1] = True
        penalize = 0.
        if direction == 0 and heroX > 1 :
            heroX -= 1
        elif direction == 1 and heroX < self.sizeX-2 :
            heroX += 1
        elif direction == 2 and heroY > 1:
            heroY -= 1
        elif direction == 3 and heroY < self.sizeY-2:
            heroY += 1
        #if hero.x == heroX and hero.y == heroY:
        #    penalize = 0.0
        if isin_([heroX-1, heroY-1], self.wall)==False:
            self.hero_[0] = heroX
            self.hero_[1] = heroY
        # self.counter = self.counter - 1
        # if self.counter <= 0 :
        #     self.locations_visited_period = self.locations_visited.copy()
        #     self.counter= np.random.randint(4,8)
        # self.update_last_locations(self.objects[0].x-1,self.objects[0].y-1)
        return penalize

    def locate_observed_feature(self,Direction):
#        hero = self.objects[0]
        X = self.hero_[0]
        Y = self.hero_[1]
        max_number_of_observations = self.sizeX*4-4 
        center = 0
        span = 0
        if Direction==0: #UP
            center = Y
            span = X*np.tan(np.pi/8)
        elif Direction==1: #DOWN
            center = self.sizeY*3-3-Y
            span = (self.sizeX-X)*np.tan(np.pi/8)
        elif Direction==2: #LEFT
            center = max_number_of_observations-X
            span = Y*np.tan(np.pi/8)
        elif Direction==3: #RIGHT
            center = self.sizeY-1+X
            span = (self.sizeY-Y)*np.tan(np.pi/8)
        output = np.arange(np.round(center-span),np.round(center+span+1))%max_number_of_observations
        return output
    def locate_observed_wall(self,Direction):
        observered_wall_position = np.zeros((self.sizeX-2,self.sizeY-2),dtype=bool)
#        hero = self.objects[0]
        X = self.hero_[0]
        Y = self.hero_[1]
        temp = [X,Y]
        #pdb.set_trace()
        if Direction==0:
            b = X
        elif Direction==1:
            b = self.sizeX-X
            # temp[0] = self.sizeX-1
        elif Direction==2:
            b = Y
            #temp[1] = 1
        elif Direction==3:
            b = self.sizeY-Y
            # temp[1] = self.sizeY-1
        #print(Direction)
        for i in range(0,b+1):
            a = int(np.round((b-i) * np.tan(np.pi/8.)))
#            temp_observed_wall =[]
            for j in range(0,a+1):
                if (Direction==0):
#                    pdb.    ()
                    temp[0] = i
                    temp[1] = Y-1-j
                    if isin_(temp,self.wall):
                        observered_wall_position[temp[0],temp[1]]=True
#                        temp_observed_wall.append([temp[0],temp[1]])
                    temp[1] = Y-1+j
                    if isin_(temp,self.wall):
                        observered_wall_position[temp[0],temp[1]]=True
#                        temp_observed_wall.append([temp[0],temp[1]])
                    #print(temp)
                elif(Direction==1):
                    temp[0] = X-1+b-i
                    temp[1] = Y-1-j
                    if isin_(temp,self.wall):
                        observered_wall_position[temp[0],temp[1]]=True
                    temp[1] = Y-1+j
                    if isin_(temp,self.wall):
                        observered_wall_position[temp[0],temp[1]]=True
                elif (Direction==2):
                    temp[0] = X-1-j
                    temp[1] = i
                    if isin_(temp,self.wall):
                        observered_wall_position[temp[0],temp[1]]=True
                    temp[0] = X-1+j
                    if isin_(temp,self.wall):
                        observered_wall_position[temp[0],temp[1]]=True
                    #print(temp)
                elif (Direction==3):
                    temp[0] = X-1-j
                    temp[1] = Y-1+b-i
                    if isin_(temp,self.wall):
                        observered_wall_position[temp[0],temp[1]]=True
                    temp[0] = X-1+j
                    if isin_(temp,self.wall):
                        observered_wall_position[temp[0],temp[1]]=True
        return observered_wall_position
    def locate_observed_item(self,Direction):
        observered_item_position = np.zeros((self.sizeX-2,self.sizeY-2),dtype=int)
#        hero = self.objects[0]
        X = self.hero_[0]
        Y = self.hero_[1]
        temp = [X,Y]
        #pdb.set_trace()
        if Direction==0:
            b = X
        elif Direction==1:
            b = self.sizeX-X
            # temp[0] = self.sizeX-1
        elif Direction==2:
            b = Y
            #temp[1] = 1
        elif Direction==3:
            b = self.sizeY-Y
            # temp[1] = self.sizeY-1
        #print(Direction)
        for i in range(0,b+1):
            a = int(np.round((b-i) * np.tan(np.pi/8.)))

            for j in range(0,a+1):                
                if (Direction==0):
                    temp[0] = i
                    temp[1] = Y-1-j
                    condition = isin_index(temp,self.item)
                    if condition>=0:
                        observered_item_position[temp[0],temp[1]]=self.item_id[condition]

                    temp[1] = Y-1+j
                    condition = isin_index(temp,self.item)
                    if condition>=0:
                        observered_item_position[temp[0],temp[1]]=self.item_id[condition]

                    
                elif(Direction==1):
                    temp[0] = X-1+b-i
                    temp[1] = Y-1-j
                    condition = isin_index(temp,self.item)
                    if condition>=0:
                        observered_item_position[temp[0],temp[1]]=self.item_id[condition]
                    temp[1] = Y-1+j
                    condition = isin_index(temp,self.item)
                    if condition>=0:
                        observered_item_position[temp[0],temp[1]]=self.item_id[condition]
                elif (Direction==2):
                    temp[0] = X-1-j
                    temp[1] = i
                    condition = isin_index(temp,self.item)
                    if condition>=0:
                        observered_item_position[temp[0],temp[1]]=self.item_id[condition]
                    temp[0] = X-1+j
                    condition = isin_index(temp,self.item)
                    if condition>=0:
                        observered_item_position[temp[0],temp[1]]=self.item_id[condition]
                    #print(temp)
                elif (Direction==3):
                    temp[0] = X-1-j
                    temp[1] = Y-1+b-i
                    condition = isin_index(temp,self.item)
                    if condition>=0:
                        observered_item_position[temp[0],temp[1]]=self.item_id[condition]
                    temp[0] = X-1+j
                    condition = isin_index(temp,self.item)
                    if condition>=0:
                        observered_item_position[temp[0],temp[1]]=self.item_id[condition]
        return observered_item_position
    
    def step(self, action, direction =-1):
        self.last_reward_and_action.pop(0)
        reward, done = self.checkGoal(0,action)
        if direction == -1:                
            direction = np.random.randint(0,4)
        self.moveChar(direction)
        state = self.get_screen(direction)#
        self.last_reward_and_action.append([reward,action])        
        return state, reward, done, direction
    def checkGoal(self,direction,action):
        Complete = False
        # visited = 0
        x = self.hero_[0]-1
        y = self.hero_[1]-1
        reward = 0
        # print([x-1,y-1],self.LC_loc,'\n',isin_index([x-1,y-1],self.LC_loc))
        if (self.LC_[x,y]==True):
            if (self.locations_visited[x,y]==True):                
                if action >0:#== isin_index([x,y],self.LC_loc)+1:
                    reward = 100
                    self.total_reward = self.total_reward + reward
                    self.visit_count += 1
                else:
                    reward = -100
            elif action == 0:
                reward = 10
                self.total_reward = self.total_reward + reward
            else:
                reward = -100
                # Complete = True
                # self.total_reward = self.total_reward + reward
                # print(self.total_reward)
                # return reward, Complete
        elif action > 0:
            reward = -1
            self.total_reward = self.total_reward + reward

        elif action== 0:
            reward = 10
            self.total_reward = self.total_reward + reward
    
        if (np.sum(self.locations_visited.flatten())==(((self.sizeX-2)**2)//1.1 -len(self.wall) ) ):
            Complete = True
            if self.visit_count>3: #>=len(self.temp_x):
                reward = 1000            
            # print(self.total_reward)            
        return reward, Complete

    def get_screen(self,direction=0,res=84):
        a = np.ones([self.sizeY,self.sizeX,3], dtype=np.uint8)
        a[1:-1,1:-1,:] = 0
#        hero = self.objects[0]
        # observations = self.locate_observed_feature(direction)
        walls = self.locate_observed_wall(direction)
        items = self.locate_observed_item(direction)
        
        start_x, start_y = 0, 0
        end_x, end_y = 0, 0

        if direction ==0:
            start_x, start_y = 0, self.hero_[1]#self.sizeY-2
            end_x, end_y = self.sizeX-1, 0
        elif direction ==1:
            start_x, start_y = 0, 0
            end_x, end_y = self.sizeX-1, self.sizeY-2
        elif direction ==2:
            start_x, start_y = self.sizeX-2, 0
            end_x, end_y = 0, self.sizeY-1
        elif direction ==3:
            start_x, start_y = 0, 0
            end_x, end_y = self.sizeX-2 , self.sizeY-1
        
        for i in range(1,self.sizeX-1):
            for j in range(1,self.sizeY-1):
                if walls[i-1,j-1]==True:
                    a[i,j,:] = 200
                if items[i-1,j-1]==True:
                    a[i,j,:] = items[i-1,j-1]
                #a[i,j,:] = self.locations_visited[i-1,j-1]*127
#        for item in self.objects:
##             if item.intensity in observations:
##                 print(item.intensity)
#            # if item.ID in observations:
#            #     a[item.x:item.x+item.size,item.y:item.y+item.size,item.channel] = item.intensity  
#            # el
#            if item.name == 'hero':
#                a[item.x:item.x+item.size,item.y:item.y+item.size,:] = 255
#        # a = resize(a,(self.res, self.res,3),order=0,mode='wrap',anti_aliasing=True)
#        a[hero.x,hero.y,:]= [255, 242, 0]
        return a #resize_T(a).unsqueeze(0).to(device)
   
    def get_env_with_path(self,res=40):
        a = np.ones([self.sizeY,self.sizeX,3],dtype=np.uint8)
        #a[1:-1,1:-1,:] = 255
#        hero = self.objects[0]
        for item in self.objects:
            a[item.x:item.x+item.size,item.y:item.y+item.size,item.channel] = item.intensity
#            if item.name == 'hero':
#                hero = item
        for i in range(1,self.sizeX-1):
            for j in range(1,self.sizeY-1):
                if isin_([i-1,j-1],self.wall):
                    a[i,j,:] = 255
                elif self.LC_[i-1,j-1]==True:
                    a[i,j,:] = 155
                elif self.locations_visited[i-1,j-1]==True:
                    a[i,j,:] = a[i,j,:] + 55
                elif isin_index([i-1,j-1],self.item)>=0:
                    a[i,j,:] = self.item_id[isin_index([i-1,j-1],self.item)]
                else:
                    a[i,j,:] = 0
        a[self.hero_[0],self.hero_[1],:]= [255, 242, 0]#a[hero.x,hero.y,:]+100
        #res = 84
        # a = resize(a,(self.res,self.res,3),order=0,mode='wrap',anti_aliasing=True)
        return a #resize_T(a).unsqueeze(0).to(device)  
    def current_loc(self):
        return self.hero_
