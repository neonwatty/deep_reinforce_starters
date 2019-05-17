import matplotlib.pyplot as plt
import numpy as np

# moving average function
def moving_average(x,D):
    y = []
    for p in range(len(x)+1):
        # make next element
        b = np.sum(x[np.maximum(0,p-D):p])/float(D)
        y.append(b)
    return np.array(y)

def plot_reward_history(logname,**kwargs):
    start = 1
    window_length = 5
    if 'window_length' in kwargs:
        window_length = kwargs['window_length']
    if 'start' in kwargs:
        start = kwargs['start']
        
    # load in total episode reward history
    data = np.loadtxt(logname)
    ave = moving_average(data,window_length)

    # create figure
    fig = plt.figure(figsize = (12,8))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)

    # plot total reward history
    ax1.plot(data)
    ax1.set_xlabel('episode',labelpad = 8,fontsize = 13)
    ax1.set_ylabel('total reward',fontsize = 13)
    
    ave[:window_length] = np.nan
    ax2.plot(ave,linewidth=3)
    ax2.set_xlabel('episode',labelpad = 8,fontsize = 13)
    ax2.set_ylabel('ave total reward',fontsize=13)
    plt.show()
    

def compare_reward_histories(lognames,**kwargs):
    # create figure
    fig = plt.figure(figsize = (12,8))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    
    for name in lognames:
        # load in total episode reward history
        data = np.loadtxt(name)
        ave = [data[v] for v in range(100)]

        for i in range(0,np.size(data)-100):
            m = np.mean(data[i:i+100])
            ave.append(m)

        # plot total reward history
        start = 0
        if 'start' in kwargs:
            start = kwargs['start']
        ax1.plot(data[start:])
        ax1.set_xlabel('episode',labelpad = 8,fontsize = 13)
        ax1.set_ylabel('total reward',fontsize = 13)

        ax2.plot(ave[start:],linewidth=3)
        ax2.set_xlabel('episode',labelpad = 8,fontsize = 13)
        ax2.set_ylabel('ave total reward',fontsize=13)
    plt.show()