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
    def __init__(self,size,_lc_count=4,_wall_count=4,_seed_id=0,res=84):#,seed_ = np.random.randint(1000)):
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
        self.total_reward=0
        self.last_location = []
        self.last_location_buffer_size = 8
        self.locations_visited = np.zeros((size,size),dtype=np.bool)
        self.LC_ = self.locations_visited.copy()
        # self.locations_visited_period = self.locations_visited.copy()
        # self.last_locations =  np.zeros((3,2),dtype=np.uint16)
        # self.temp_y= [1,2,self.sizeX-5,self.sizeX-4]#[6,7,8,9,10,11,12,13,6,7,8,9,10,11,12,13]#line wall :[1,2,self.sizeX-5,self.sizeX-4,5,6,1,2]
        # self.temp_x= [1,2,self.sizeX-5,self.sizeX-4]#[5,5,5,5,5,5,5,5,15,15,15,15,15,15,15,15]#line wall :[1,2,self.sizeX-5,self.sizeX-4,1,2,self.sizeX-5,self.sizeX-4]
        self.visit_count = 0
        self.wall  = []#[[3,3],[3,4],[3,5],[3,6],[3,7],[4,3],[5,3],[6,3],[7,3],[5,7],[6,7],[7,7],[7,5],[7,6]]
        # for i in range(_wall_count):
        i=0
        # while(i<_wall_count):
        #     temp_x = np.random.randint(0,size)
        #     temp_y = np.random.randint(0,size)
        #     try:
        #         if np.isin( ((np.array(self.wall[:,0]),temp_x) * np.isin(np.array(self.wall[:,1]),temp_y)).any()): 
        #             continue
        #         else:
        #             self.wall.append([temp_x,temp_y])
        #             i=i+1
        #     except:
        #         self.wall.append([temp_x,temp_y])
        #         i=i+1
        #for loop in range(): #np.random.randint(4,self.sizeX-2)
            #self.temp_x.append()#np.random.randint(0,self.sizeX-2))
            #self.temp_y.append()#np.random.randint(0,self.sizeY-2))
        # for loop in range(len(self.temp_x)):#3,self.sizeX-5):
            # self.wall.append([self.temp_x[loop],self.temp_y[loop]])
            #self.wall.append([7,loop])
        for loop in range(3,7):
            self.wall.append([loop,3])
            self.wall.append([loop,4])
            self.wall.append([loop,5])
            self.wall.append([loop,6])
        # for i in range(2,8):
        #     # for j in range(3,7):
        #     self.wall.append([2,i])
        # for i in range(2,8):
        #     # for j in range(3,7):
        #     self.wall.append([i,2])
        # for i in range(5,8):
        #     # for j in range(3,7):
        #     self.wall.append([7,i])
        # for i in range(5,8):
        #     # for j in range(3,7):
        #     self.wall.append([i,8])
        # self.wall.append(([3,6],[3,6],[6,3],[6,6]))
        # self.wall.extend(([3,3],[3,6],[3,6],[6,3],[6,6]))

        self.temp_x= [2,2,7,7]#,4,5,4,5]#,1,1,2]
        self.temp_y= [2,7,2,7]#,4,4,5,5]#,1,2,1]
        self.LC_loc= []
        # self.seed(self.seed_id)    
        self.counter = np.random.randint(2,6)
        newPoistion = [np.random.randint(1,self.sizeX-1),np.random.randint(1,self.sizeY-1)]
        # print(newPoistion,'\n',self.wall)
        while ([newPoistion[0]-1,newPoistion[1]-1],self.wall)==True:
            # print(newPoistion)
            newPoistion = [np.random.randint(1,self.sizeX-1),np.random.randint(1,self.sizeY-1)]
        print(newPoistion)
        hero = gameOb(newPoistion,1,10,0,None,'hero',-1)
        self.objects.append(hero)
        points = []
        color_list = []
        for i in range(self.sizeX):
            points.append([0,i])
        for i in range(1,self.sizeX):
            points.append([i,self.sizeX-1])
        for i in range(self.sizeX-2,-1,-1):
            points.append([self.sizeX-1,i])
        for i in range(self.sizeX-2,0,-1):
            points.append([i,0])
        #selecting the seed for random number
        # self.seed(self.seed_id) 
        channel = np.random.randint(0,3,self.sizeX*4-4)
        color = np.random.randint(5,255,self.sizeX*4-4)
        for i in range(self.sizeX*4-4): 
            # channel = 2#np.random.randint(0,3)
            # color = i*3#np.random.randint(1,250)
            # while (np.sum(np.isin(color_list,color[i]))!=0):
            #     color[i] = np.random.randint(5,255)
            color_list.append(color[i])
            feature = gameOb(points[i],1,color[i],channel[i],None,'feature'+str(i),i)
            self.objects.append(feature)
        
        

        self.actions=[0]
        for loop in range(len(self.temp_x)):
            self.LC_[self.temp_x[loop],self.temp_y[loop]]=True
            self.LC_loc.append([self.temp_x[loop],self.temp_y[loop]])
            self.actions.append(loop+1)
        i=0
        # while(i<_lc_count):
        #     temp_x = int(np.random.randint(0,size))
        #     temp_y = int(np.random.randint(0,size))
        # #     # print(temp_x,temp_y)

        #     try:
        #         cond1 = np.isin( ((np.array(self.LC_loc[:,0]),temp_x) * np.isin(np.array(self.LC_loc[:,1]),temp_y)).any())
        #         cond2 = np.isin( ((np.array(self.wall[:,0]),temp_x) * np.isin(np.array(self.wall[:,1]),temp_y)).any())
        #         if cond1 or cond2:
        #             continue
        #         else:
        #             self.LC_[temp_x,temp_y]=True
        #             self.LC_loc.append([temp_x,temp_y])
        #             self.actions.append(i+1)
        #             i=i+1
        #     except:
        #         self.LC_[temp_x,temp_y]=True
        #         self.LC_loc.append([temp_x,temp_y])
        #         self.actions.append(i+1)
        #         i=i+1

        # print(self.LC_loc, self.wall)
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
        newPoistion = [np.random.randint(1,self.sizeX-1),np.random.randint(1,self.sizeY-1)]
        while ([newPoistion[0]-1,newPoistion[1]-1],self.wall)==True:
            newPoistion = [np.random.randint(1,self.sizeX-1),np.random.randint(1,self.sizeY-1)]
        print(newPoistion)
        # hero = gameOb(newPoistion,1,10,0,None,'hero',-1)
        self.objects[0].x = copy.deepcopy(newPoistion[0])
        self.objects[0].y = copy.deepcopy(newPoistion[1])
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
        heroX = self.objects[0].x
        heroY = self.objects[0].y
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
            self.objects[0].x = heroX
            self.objects[0].y = heroY
        # self.counter = self.counter - 1
        # if self.counter <= 0 :
        #     self.locations_visited_period = self.locations_visited.copy()
        #     self.counter= np.random.randint(4,8)
        # self.update_last_locations(self.objects[0].x-1,self.objects[0].y-1)
        return penalize

    def locate_observed_feature(self,Direction):
        hero = self.objects[0]
        X = hero.x
        Y = hero.y
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
        hero = self.objects[0]
        X = hero.x
        Y = hero.y
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
            temp_observed_wall =[]
            for j in range(0,a+1):
                if (Direction==0):
#                    pdb.set_trace()
                    temp[0] = i
                    temp[1] = Y-1-j
                    if isin_(temp,self.wall):
                        observered_wall_position[temp[0],temp[1]]=True
                        temp_observed_wall.append([temp[0],temp[1]])
                    temp[1] = Y-1+j
                    if isin_(temp,self.wall):
                        observered_wall_position[temp[0],temp[1]]=True
                        temp_observed_wall.append([temp[0],temp[1]])
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
        x = self.objects[0].x-1
        y = self.objects[0].y-1
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
        hero = self.objects[0]
        # observations = self.locate_observed_feature(direction)
        walls = self.locate_observed_wall(direction)
        start_x, start_y = 0, 0
        end_x, end_y = 0, 0

        if direction ==0:
            start_x, start_y = 0, hero.y#self.sizeY-2
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
                #a[i,j,:] = self.locations_visited[i-1,j-1]*127
        for item in self.objects:
#             if item.intensity in observations:
#                 print(item.intensity)
            # if item.ID in observations:
            #     a[item.x:item.x+item.size,item.y:item.y+item.size,item.channel] = item.intensity  
            # el
            if item.name == 'hero':
                a[item.x:item.x+item.size,item.y:item.y+item.size,:] = 255
        # a = resize(a,(self.res, self.res,3),order=0,mode='wrap',anti_aliasing=True)
        a[hero.x,hero.y,:]= [255, 242, 0]
        return a #resize_T(a).unsqueeze(0).to(device)
   
    def get_env_with_path(self,res=40):
        a = np.ones([self.sizeY,self.sizeX,3],dtype=np.uint8)
        #a[1:-1,1:-1,:] = 255
        hero = self.objects[0]
        for item in self.objects:
            a[item.x:item.x+item.size,item.y:item.y+item.size,item.channel] = item.intensity
            if item.name == 'hero':
                hero = item
        for i in range(1,self.sizeX-1):
            for j in range(1,self.sizeY-1):
                if isin_([i-1,j-1],self.wall):
                    a[i,j,:] = 255
                elif self.LC_[i-1,j-1]==True:
                    a[i,j,:] = 155
                elif self.locations_visited[i-1,j-1]==True:
                    a[i,j,:] = a[i,j,:] + 55
                else:
                    a[i,j,:] = 0
        a[hero.x,hero.y,:]= [255, 242, 0]#a[hero.x,hero.y,:]+100
        #res = 84
        # a = resize(a,(self.res,self.res,3),order=0,mode='wrap',anti_aliasing=True)
        return a #resize_T(a).unsqueeze(0).to(device)  
    def current_loc(self):
        return [self.objects[0].x, self.objects[0].y]
