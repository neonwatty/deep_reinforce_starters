# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
import autograd.numpy as np
import copy
from IPython.display import clear_output
import time

# import custom JS animator
# reinforce_lib.JSAnimation_slider_only import IPython_display_slider_only

# import openai gym
import gym

class Animate:
    def __init__(self,simulator,learner,**kwargs):
        # initialize enviroment
        self.simulator = simulator
        
        # load in qlearner
        self.learner = learner
        
        # find best set of weights to use
        max_ind = np.argmax(self.learner.training_reward)
        if 'max_ind' in kwargs:
            max_ind = kwargs['max_ind']
        
        # get associated set of weights, fix to model
        w_best = self.learner.save_weights[max_ind]
        self.Q = lambda s: self.learner.func_approx.model(s,w_best)
        
        # print update
        update = 'episode ' + str(max_ind) + ' weights chosen, where the learner achieved a total reward of ' + str(self.learner.training_reward[max_ind])
        print (update)

    def state_normalizer(self,states):
        states = np.array(states)[:,np.newaxis]
        return states
    
    # animate a single test run
    def run_episode(self,**kwargs):
        # max steps to animate
        self.max_steps = 150
        if 'max_steps' in kwargs:
            self.max_steps = kwargs['max_steps']
        
        #### start validation loop ###
        # start up simulation episode
        state = self.simulator.reset()

        # run validation, collect rendered frames
        self.frames = []
        self.actions = []
        total_reward = 0
        for k in range(self.max_steps):
            # render current state in animation
            my_img = self.simulator.render(mode='rgb_array');
            self.frames.append(my_img)
            
            # evluate all action functions on this bias-extended state
            state_array = self.state_normalizer(state)
            qvals = self.Q(state_array)
            action = np.argmax(qvals)
            self.actions.append(action)

            # take action, receive output
            new_state, reward, done, info = self.simulator.step(action)
            state = copy.deepcopy(new_state)
            total_reward += reward
            # exit this episode if complete
            if done:
                # close the pop up window renderer
                self.simulator.render(close=True)
                break
        self.simulator.render(close=True)
        
        # print number of steps taken in this episode
        update = 'simulator ran for ' + str(len(self.frames)) + ' steps'
        print (update)
        update = 'total reward for episode = ' + str(total_reward) 
        print (update)

    def animate_run(self):        
        # initialize figure
        fig = plt.figure(figsize = (10,5))
        artist = fig

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 1) 
        ax = plt.subplot(gs[0]); ax.set_aspect('equal')
        num_frames = len(self.frames)
        def animate_it(k):
            # clear panel
            ax.cla()
            
            # print rendering update            
            if np.mod(k+1,10) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
                
            ax.imshow(self.frames[k])
            title = 'step ' + str(k+1) + ', action = ' + str(self.actions[k])
            ax.set_title(title,fontsize = 18)
            return artist,
        
        anim = animation.FuncAnimation(fig, animate_it,frames=len(self.frames), interval=len(self.frames), blit=True)
        return(anim)