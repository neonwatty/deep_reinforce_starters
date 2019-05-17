import numpy as np
import copy
import os
import pickle
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.models import Sequential
import gym

class QLearner():
    # load in simulator, initialize global variables
    def __init__(self,dirname,savename,**kwargs):
        # make simulator global
        self.simulator = gym.make('CartPole-v1') 
        self.simulator = gym.make('LunarLander-v2') 
             

        # Q learn params
        self.explore_val = 1
        self.explore_decay = 0.99
        self.num_episodes = 500        
        self.gamma = 1
        self.episode_update = 10
        self.exit_level = 200
        self.exit_window = 100
        
        if "gamma" in kwargs:   
            self.gamma = args['gamma']
        if 'explore_val' in kwargs:
            self.explore_val = kwargs['explore_val']
        if 'explore_decay' in kwargs:
            self.explore_decay = kwargs['explore_decay']
        if 'num_episodes' in kwargs:
            self.num_episodes = kwargs['num_episodes']
        if 'episode_update' in kwargs:
            self.episode_update = kwargs['episode_update']
        if 'exit_level' in kwargs:
            self.exit_level = kwargs['exit_level']
        if 'exit_window' in kwargs:
            self.exit_window = kwargs['exit_window']   
            
        # other training variables
        self.num_actions = self.simulator.action_space.n
        state = self.simulator.reset()    
        self.state_dim = np.size(state)
        self.training_reward = []
        
        # setup memory params
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
        self.reward_logname = dirname + '/reward_logs/' + savename + '.txt'
        self.weight_name = dirname + '/saved_model_weights/' + savename + '.pkl' 
        self.model_name = dirname + '/models/' + savename + '.json'

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
        lr = 10**(-2)
        if 'alpha' in kwargs:
            lr = kwargs['alpha']
        optimizer = RMSprop(lr = lr)        

        # input / output sizes of network
        input_dim = self.state_dim
        output_dim = self.num_actions

        # build model based on parameters
        self.model = Sequential()

        # add input layer
        self.model.add(Dense(layer_sizes[0], input_dim=input_dim, activation=activation))

        # add hidden layers
        for U in layer_sizes[1:]:
            self.model.add(Dense(U,activation=activation))

        # add output layer
        self.model.add(Dense(output_dim, activation='linear'))

        # chose optimizer and its associated parameters
        self.model.compile(loss=loss, optimizer=optimizer)
        
        # initialize Q
        self.Q = self.model.predict
        
        # since its easy to save models with keras / tensorflow, save to file
        model_json = self.model.to_json()
        with open(self.model_name, "w") as json_file:
            json_file.write(model_json)

    # update Q function
    def update_Q(self):        
        # generate q_values based on most recent Q
        q_vals = []
        states = []
        for i in range(len(self.memory)):    
            # get episode_data
            episode_data = self.memory[i]
            
            # loop over episode data and create input/output pairs
            for j in range(len(episode_data)):
                # get next sample of episode
                sample = episode_data[j]
            
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
                    qs = self.Q(next_state)
                    q += self.gamma*np.max(qs)

                # clamp all other models to their current values for this input/output pair
                q_update = self.Q(state).flatten()
                q_update[action] = q
                q_vals.append(q_update)
                states.append(state.T)
            
        # convert lists to numpy arrays for regressor
        s_in = np.array(states).T
        q_vals = np.array(q_vals).T
        s_in = s_in[0,:,:]
                            
        # take descent step
        num_pts = s_in.shape[1]
        self.model.fit(x=s_in.T,y=q_vals.T,batch_size=num_pts, epochs=1,verbose = 0)
        
        # update Q based on regressor updates
        self.Q = self.model.predict
        
    ##### functions for adjusting replay memory #####
    # update memory - add sample to list, remove oldest samples 
    def update_memory(self,episode_data):
        # add most recent trial data to memory
        self.memory.append(episode_data)

        # clip memory if it gets too long    
        num_episodes = len(self.memory)
        if num_episodes >= self.memory_length:    
            num_delete = num_episodes - self.memory_length
            self.memory[:num_delete] = []
    
    ##### Q Learning functionality #####
    # special function to check done
    def check_done(self,done,reward):
        if done == True:
            reward = -100
        return done,reward
    
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
        qs = self.Q(state) 
        if p > self.explore_val:
            action = np.argmax(qs)
        return action
    
    # main training function
    def train(self,**kwargs):             
       
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
            episode_data = []
            while done == False and step < 500:    
                # choose next action
                action = self.choose_action(state)
    
                # transition to next state, get associated reward
                next_state,reward,done,info = self.simulator.step(action)  
                next_state = self.state_normalizer(next_state)
                
                # store data for transition after episode ends
                episode_data.append([state,next_state,action,reward,done])

                # update total reward from this episode
                total_episode_reward+=reward
                state = copy.deepcopy(next_state)
                step+=1
            
            # update memory with this episode's data
            self.update_memory(episode_data)
            
            # update Q function
            if np.mod(n,self.episode_update) == 0:
                self.update_Q()  
                  
            # update episode reward greater than exit_level, add to counter
            exit_ave = np.sum(np.array(self.training_reward[-self.exit_window:]))/self.exit_window

            ### print out updates ###
            update = 'episode ' + str(n+1) +  ' of ' + str(self.num_episodes) + ' complete, ' +  ' explore val = ' + str(np.round(self.explore_val,3)) + ', episode reward = ' + str(np.round(total_episode_reward,1)) + ', ave reward = ' + str(np.round(exit_ave,1))
                              
            print (update)

            self.update_log(self.logname,update + '\n')

            # if the total episode reward is greater than exit level for 10 consecutive episodes, break
            if exit_ave >= self.exit_level:
                break
                              
            update = str(total_episode_reward) + '\n'
            self.update_log(self.reward_logname,update)

            ### store this episode's computation time and training reward history
            self.training_reward.append(total_episode_reward)

            # save latest weights from this episode 
            update = self.model.get_weights()
            self.update_log(self.weight_name,update)
            
        ### save weights ###
        update = 'q-learning algorithm complete'
        self.update_log(self.logname,update + '\n')
        print (update)