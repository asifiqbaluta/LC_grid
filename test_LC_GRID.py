from LC_GRID_fixed import *
import sys
from PIL import Image
import pygame
from mpl_toolkits.mplot3d import axes3d

#############
print("Testing Environment...\n")
plt.ion()
pygame.init()
screen = pygame.display.set_mode((320,240))
clock = pygame.time.Clock()

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
plt.figure(1)
ax1.axis('off')
ax2.axis('off')
reward = 0
action = 0
done = False
#action_list = ['up','down','left','right','no action']
direction_list = ['up','down','left','right']
action_list = ['NL','LC'] #['no action','loop closure']
env = gameEnv(size=10,_lc_count=5,_wall_count=5,_seed_id=11)#,res=12)
cm=0
m = 0
m_count = 20
res = 84
l=0
total_reward = 0
# env.seed(10000+2)
finished = False
#plt.switch_backend('TkAgg')
np.random.seed(225)
ts=[]
lcs=[]
stop_self_play = True
pygame_locs= []
for enn_l in range(1):
    env.reset()
    total_step = 0 
    lc_step = 0
    seed = np.random.random()
    print(int(seed*1000))
    env.seed(int(seed*1000)+1)
    # print(env.objects[1].intensity)
    # for l in range(200):
    #     action = np.random.randint(0,64)
    #     _, reward, done, direction  = env.step(action, cm, m_count)
    #     total_reward = total_reward + reward
    # for m in range(50):
    while (finished == False):
        #env.step(action,cm,m_count)

        # get pressed key 
        event = pygame.event.poll()
        if event.type == pygame.QUIT:
            finished = True
        if event.type == pygame.KEYDOWN:
            if event.key==pygame.K_UP: 
                direction = 0
            elif event.key==pygame.K_LEFT: 
                direction = 2
            elif event.key==pygame.K_RIGHT:
                direction = 3
            elif event.key==pygame.K_DOWN:
                direction = 1
            elif event.key==pygame.K_TAB:
                direction=-1
            elif event.key==pygame.K_SPACE:
                stop_self_play = False
                    
            if stop_self_play:
                # action = np.random.randint(0,len(env.actions))
                action = 0
            else:
                input_ = input("enter action")
                action = int(input_)
                stop_self_play = True
                
            if direction==-1:
                for i in range(50):
                    env.step(0,-1)

            # action = 0
            _, reward, done, direction  = env.step(action,direction)
            total_reward = total_reward + reward
            
            print(direction,action,reward)
            

            im_step = env.get_screen(direction)
            im = env.get_env_with_path()
            ax1.imshow(im) #[:,:,0],cmap='gray')
            ax2.imshow(im_step) #[:,:,0],cmap = 'gray')
            ax2.set_title([reward,total_reward])
            ax1.set_title([env.objects[0].x-1, env.objects[0].y-1])
            # im = Image.fromarray(im_step)
            # im.save("output/step/"+str(m)+"_step.png")
            # im = Image.fromarray(env.get_env_with_path())
            # im.save("output/env/"+str(m)+"_env.png")
            m = m+1
            # if m%50==0:
            #     print(m)            
            plt.show()
            plt.pause(0.1)
            if done:
                print(done)
                pygame.quit()
                input("Press to exit:")
                break
            pygame.display.flip()
        #plt.savefig('output/plots/im_'+str(m)+'.jpg')

    # for i in range(5000):
#         #im = env.get_env_with_path(res)[:,:,0]
#         #im_step  = env.get_screen(res)[:,:,0]
#         #ax1.imshow(im,cmap='gray')  
#         #ax1.set_title([action_list[action], reward, done])
        # direction = 0
        # action = 1
#         #for s in range(0,1):
#             #im_step  = env.get_screen(direction,40)[:,:,0]
#             #im_step = env.get_env_with_path(res)[:,:,0]
#             #ax2.imshow(im_step, cmap = 'gray')
#             #ax2.set_title([action_list[action],reward, direction_list[direction]])
#             #_, reward, done, direction,m_count,_ = env.step(action,cm,m_count)
#             #plt.show()
#             #plt.pause(0.5)
#         #for j in range(4):
#             #direction = np.random.randint(0,4)
#             #env.step(action,cm,m_count)
#             #im_step = env.get_screen(direction,res)[:,:,0]
#             #ax2.imshow(im_step,cmap = 'gray')
#             #plt.savefig('output/im_'+str(i)+'_'+str(j)+'.jpg')
#         direction = -1 
                                           
#         while (direction < 0) or (direction > 3):
#             direction = input("Press between 0 to 3 to continue: ")
#             #print(input_)
#             #print((input_!= '0') or (input_!='1'))
#             direction= int(direction) 
                                           
#         input_ = -1
#         while (input_!= '0') and (input_!='1'):
#             input_ = input("Press between 0 to 1 to continue: ")
#             if input_ == '5':
#                 for l in range(50):
#                     _, reward, done, direction, m_count,_ = env.step(action,cm,m_count)
#                     im_step = env.get_screen(direction,res)[:,:,0]
#                     im = env.get_env_with_path(res)[:,:,0]
#                     ax1.imshow(im,cmap='gray')
#                     ax2.imshow(im_step,cmap = 'gray')
#                     plt.show()
#                     plt.pause(0.05)
#                     # plt.savefig('output/im_'+str(i)+'_'+str(l)+'.jpg')
#         action = int(input_)
        # action = np.random.randint(0,2)
#         direction = np.random.randint(0,4)
# #         print(str(action_list[action]) +'   '+ str(direction_list[direction]) )
#         action = np.random.randint(0,len(env.actions))
#         _, reward, done, direction  = env.step(action,direction)
#         total_step = total_step+1
#         if reward ==100 or reward==-100:
#             lc_step = lc_step+1
        # _, reward, done, direction, m_count,_ = env.step(action,cm,m_count,5,direction)
#         #if done:
#             #print("reached final step ",i)
#             #break
        
#         im = env.get_env_with_path(res)[:,:,0]
#         im_step = env.get_screen(direction,res)[:,:,0]
#         #print(str(direction_list[direction])+'---------------\n')
#         #print(env.locate_observed_wall(direction))
#         title = 'Current Action: '+ str(action_list[action]) +'\n Reward: '+str(reward) #, direction_list[direction]]
#         #ax1.set_title(title)
#         ax1.imshow(im,cmap='gray')
#         ax2.imshow(im_step,cmap = 'gray')
#         #[action_list[action],reward, direction_list[direction]]
#         if reward >1:
#             reward = int(input("reward: ")
# )
#         title = 'Current Action: '+ str(action_list[action]) +'\n Reward: '+str(reward)+'\n'+ str(direction_list[direction]) #, direction_list[direction]]
#         ax2.set_title(title)
#         #plt.imshow(im,cmap = 'gray')
#         # plt.show()
#         # plt.pause(0.2)
#         # plt.savefig('output/im_'+str(i)+'.jpg')
#         plt.ion()
        #input("Press Enter to continue...")
# plt.close('all')

    # print("step: ",total_step," lc: ",lc_step)
    # ts.append(total_step)
    # lcs.append(lc_step)
# pygame.quit()
# print(np.mean(lcs),np.std(lcs))
# history = np.load('lc_action_history.npy')
# n_lc_action_mean = history[:,-2]*100/history[:,-3]
# n_lc_action_mean = np.append(n_lc_action_mean,[ 2.40151515 ,2.67407407, 3.12350427, 3.67261905,4.31037037])
# x = [4,8,12,16,20]
# grid_size = history[:,0]#[20,20,20,20,40,40,40,40,60,60,60,60,80,80,80,80]
# grid_size =np.append(grid_size,[80,80,80,80,80])
# n_lc = history[:,1]#[4,6,8,10, 8,12,16,20, 12,18,24,30, 16,24,32,40]
# n_lc = np.append(n_lc,[320,640,960,1280,1600])

# n_lc_action_mean = [1.73,1.75,1.85,1.73,  1.45,1.52,1.57,1.7,  1.3,1.35,1.41,1.44,  0.91,1.1,1.13,1.17]
# Means = [20.8,25.6,36.5,50.5,67.3]
# Std = [5,12,15,20,22]
# ind = np.arange(5)
# width = 0.35
# print(n_lc_action_mean, n_lc)
# xerr = np.random.random_sample(5)

# print(Std)

# fig, axs = plt.subplots()

# fig = plt.figure()
# x = np.arange(5)
# y = 2.5 * np.sin(x / 20 * np.pi)
# yerr = np.linspace(0.05, 0.2, 10)

# plt.errorbar(x, Means, yerr=Std,fmt='b.',markersize=5)#,linewidth=2,fillstyle ='none')
# plt.bar(ind, Means, width,yerr=Std)
# plt.xticks(ind,x)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# Grab some test data.
# X, Y, Z = axes3d.get_test_data(0.05)
# print(len(n_lc_action_mean),len(grid_size),len(n_lc))
# label_names= ['20','40','60','80']
# n_=5
# plt.rcParams.update({'font.size': 16})
# # print(n_lc.shape)
# for i in range(4):
# # Plot a basic wireframe.
#     plt.plot( n_lc[i*n_:i*n_+n_], n_lc_action_mean[i*n_:i*n_+n_],label=label_names[i])
#     # ax.scatter(grid_size[i*4:i*4+4], n_lc[i*4:i*4+4], n_lc_action_mean[i*4:i*4+4])


# plt.ylabel('Percentage of possible LC in the trajectory')
# plt.legend(title='Grid Size',fontsize='large')
# plt.xlabel('Number of LC locations')
# # ax.set_zlabel('Percentage of LC in the trajectory')
# # ax.set_zlim(0,2)

# plt.show()
print(pygame_locs)
input('enter')
##########3
plt.close('all')
