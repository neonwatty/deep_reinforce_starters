import numpy as np
import copy
import os
import pickle
import sys
import gym
from . import python_autograd_arch_designer as funclib

class QLearner():
    # load in simulator, initialize global variables
    def __init__(self,dirname,savename,**kwargs):
        # make simulator global
        self.simulator = gym.make('CartPole-v1') 
        
        # Q learn params
        self.explore_val = 1
        self.explore_decay = 0.99
        self.num_episodes = 500        
        self.gamma = 1
        
        if "gamma" in kwargs:   
            self.gamma = args['gamma']
        if 'explore_val' in kwargs:
            self.explore_val = kwargs['explore_val']
        if 'explore_decay' in kwargs:
            self.explore_decay = kwargs['explore_decay']
        if 'num_episodes' in kwargs:
            self.num_episodes = kwargs['num_episodes']
            
        # other training variables
        self.num_actions = self.simulator.action_space.n
        state = self.simulator.reset()    
        self.state_dim = np.size(state)
        self.training_reward = []
        
        # setup memory params
        self.memory_length = 2000    # length of memory replay
        self.replay_length = 200     # length of replay sample
        self.memory_start = 1000
        self.memory = []
        if 'memory_length' in kwargs:
            self.memory_length = kwargs['memory_length']
        if 'replay_length' in kwargs:
            self.replay_length = kwargs['replay_length']
        if 'memory_start' in kwargs:
            self.memory_start = kwargs['memory_start']
            
        ### initialize logs ###
        # create text file for training log
        self.logname = dirname + '/training_logs/' + savename + '.txt'
        self.reward_logname =  dirname + '/reward_logs/' + savename + '.txt'
        self.weight_name =  dirname + '/saved_model_weights/' + savename + '.pkl' 
        self.model_name =  dirname + '/models/' + savename + '.json'

        self.init_log(self.logname)
        self.init_log(self.reward_logname)
        self.init_log(self.weight_name)
        self.init_log(self.model_name)
     
    ##### logging functions #####
    def init_log(self,logname):
        # delete log if old version exists
        if os.path.exists(logname): 
            os.remove(logname)
            
    def update_log(self,logname,update):
        if type(update) == str:
            logfile = open(logname, "a")
            logfile.write(update)
            logfile.close() 
        else:
            weights = []
            if os.path.exists(logname):
                with open(logname,'rb') as rfp: 
                    weights = pickle.load(rfp)
            weights.append(update)

            with open(logname,'wb') as wfp:
                pickle.dump(weights, wfp)
    
    ##### functions for creating / updating Q #####
    def initialize_Q(self,**kwargs):
        # default parameters for network
        layer_sizes = [10,10]      # two hidden layers, 10 units each, by default
        activation = 'relu'
        if 'layer_sizes' in kwargs:
            layer_sizes = kwargs['layer_sizes']
        if 'activation' in kwargs:
            activation = kwargs['activation']

        # default parameters for optimizer - reset by hand
        loss = 'mse'
        self.lr = 10**(-2)
        if 'alpha' in kwargs:
            self.lr = kwargs['alpha']

        # input / output sizes of network
        input_dim = self.state_dim
        output_dim = self.num_actions
    
        # setup network
        layer_sizes.insert(0,input_dim)
        layer_sizes.append(output_dim)

        # setup architecture, choose cost, and setup architecture
        self.model = funclib.super_setup.Setup()
        self.model.choose_cost(name = 'least_squares')
        self.model.choose_features(layer_sizes = layer_sizes,activation = activation)
            
        # initialize Q
        self.Q = self.model.predict

    # update Q function
    def update_Q(self,state,next_state,action,reward,done):
        # add newest sample to queue
        self.update_memory(state,next_state,action,reward,done)
        
        # only update Q if sufficient memory has been collected
        if len(self.memory) < self.memory_start:
            return        
        
        # update memory sample
        self.sample_memory()
        
        # generate q_values based on most recent Q
        q_vals = []
        states = []
        for i in range(len(self.replay_samples)):    
            # get sample
            sample = self.replay_samples[i]
            
            # strip sample for parts
            state = sample[0]
            next_state = sample[1]
            action = sample[2]
            reward = sample[3]
            done = sample[4]
                            
            ### for cartpole only - check if done, and alter reward to improve learning ###
            done,reward = self.check_done(done,reward)

            # compute and store q value
            q = reward 
            if done == False:
                qs = self.Q(next_state.T)
                q += self.gamma*np.max(qs)
            
            # clamp all other models to their current values for this input/output pair
            q_update = self.Q(state.T).flatten()
            q_update[action] = q
            q_vals.append(q_update)
            states.append(state.T)
            
        # convert lists to numpy arrays for regressor
        s_in = np.array(states).T
        q_vals = np.array(q_vals).T
        s_in = s_in[0,:,:]
                            
        # take descent step
        self.model.fit(s_in,q_vals,algo = 'RMSprop',max_its = 1,alpha = self.lr,verbose = False)
        
        # update Q based on regressor updates
        self.Q = self.model.predict
        
    ##### functions for adjusting replay memory #####
    # update memory - add sample to list, remove oldest samples 
    def update_memory(self,state,next_state,action,reward,done):
        # add most recent trial data to memory
        self.memory.append([state,next_state,action,reward,done])

        # clip memory if it gets too long    
        num_elements = len(self.memory)
        if num_elements >= self.memory_length:    
            num_delete = num_elements - self.memory_length
            self.memory[:num_delete] = []
    
    # sample from memory and create input / output pairs for regression
    def sample_memory(self):
        # indices to sample
        memory_num = len(self.memory)
        sample_nums = np.random.permutation(memory_num)[:self.replay_length]

        # create samples
        self.replay_samples = [self.memory[v] for v in sample_nums]
    
    ##### Q Learning functionality #####
    # state normalizer
    def state_normalizer(self,states):
        states = np.array(states)[np.newaxis,:]
        return states
    
    # choose next action
    def choose_action(self,state):
        # pick action at random
        p = np.random.rand(1)   
        action = np.random.randint(self.num_actions)
            
        # pick action based on exploiting
        if len(self.memory) >= self.memory_start:
            qs = self.Q(state.T) 
            if p > self.explore_val:
                action = np.argmax(qs)
        return action

    # special function to check done
    def check_done(self,done,reward):
        if done == True:
            reward = -100
        return done,reward
    
    # main training function
    def train(self,**kwargs):   
        verbose = True
        if 'verbose' in kwargs:
            verbose = kwargs['verbose']
            
        ### start main Q-learning loop ###
        for n in range(self.num_episodes): 
            # pick this episode's starting position - randomly initialize from f_system
            state = self.simulator.reset()    
            state = self.state_normalizer(state)
            total_episode_reward = 0
            done = False
            
            # get out exploit parameter for this episode
            if self.explore_val > 0.01:
                self.explore_val *= self.explore_decay
                    
            # run episode
            step = 0
            while done == False and step < 500:    
                # choose next action
                action = self.choose_action(state)
    
                # transition to next state, get associated reward
                next_state,reward,done,info = self.simulator.step(action)  
                next_state = self.state_normalizer(next_state)
                
                # update Q function
                self.update_Q(state,next_state,action,reward,done)  

                # update total reward from this episode
                total_episode_reward+=reward
                state = copy.deepcopy(next_state)
                step+=1
                  
            # print out update if verbose set to True
            update = 'training episode ' + str(n+1) +  ' of ' + str(self.num_episodes) + ' complete, ' +  ' explore val = ' + str(np.round(self.explore_val,3)) + ', episode reward = ' + str(np.round(total_episode_reward,2)) 

            self.update_log(self.logname,update + '\n')
                
            # print out update
            if verbose == True:
                print (update)

            update = str(total_episode_reward) + '\n'
            self.update_log(self.reward_logname,update)

            ### store this episode's computation time and training reward history
            self.training_reward.append(total_episode_reward)

            # save latest weights from this episode 
            update = self.model.weight_history[-1]
            self.update_log(self.weight_name,update)
            
        ### save weights ###
        update = 'q-learning algorithm complete'
        self.update_log(self.logname,update + '\n')
        print (update)