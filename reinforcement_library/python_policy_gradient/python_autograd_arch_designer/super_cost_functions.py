import autograd.numpy as np
from inspect import signature

class Setup:
    def __init__(self,name,**kwargs):        
        ### make cost function choice ###
        # for regression
        if name == 'least_squares':
            self.cost = self.least_squares
        if name == 'least_absolute_deviations':
            self.cost = self.least_absolute_deviations
            
        # for multiclass classification
        if name == 'multiclass_perceptron':
            self.cost = self.multiclass_perceptron
        if name == 'multiclass_softmax':
            self.cost = self.multiclass_softmax
            
        # for two-class classification
        if name == 'softmax':
            self.cost = self.softmax
        if name == 'perceptron':
            self.cost = self.perceptron
            
    ### insert feature transformations to use ###
    def define_feature_transform(self,feature_transforms):
        # make copy of feature transformation
        self.feature_transforms = feature_transforms
        
        # count parameter layers of input to feature transform
        self.sig = signature(self.feature_transforms)
            
    ##### models functions #####
    # compute linear combination of features
    def model(self,x,w):   
        # feature transformation - switch for dealing
        # with feature transforms that either do or do
        # not have internal parameters
        f = 0
        if len(self.sig.parameters) == 2:
            f = self.feature_transforms(x,w[0])
        else: 
            f = self.feature_transforms(x)    

        # compute linear combination and return
        # switch for dealing with feature transforms that either 
        # do or do not have internal parameters
        a = 0
        if len(self.sig.parameters) == 2:
            a = w[1][0] + np.dot(f.T,w[1][1:])
        else:
            a = w[0] + np.dot(f.T,w[1:])
        return a.T

    ###### regression costs #######
    # an implementation of the least squares cost function for linear regression
    def least_squares(self,w,x,y,beta,iter):
        # get batch of points
        x_p = x[:,iter]
        y_p = y[:,iter]
        beta_p = beta[:,iter]
                
        # compute cost over batch
        cost = np.sum((beta_p*self.model(x_p,w) - y_p)**2)
        return cost/float(np.size(y_p))

    # a compact least absolute deviations cost function
    def least_absolute_deviations(self,w,x,y,beta,iter):
        # get batch of points
        x_p = x[:,iter]
        y_p = y[:,iter]
        beta_p = beta[:,iter]

        # compute cost over batch
        cost = np.sum(beta_p*np.abs(self.model(x_p,w) - y_p))
        return cost/float(np.size(y_p))
 
    ###### two-class classification costs #######
    # the convex softmax cost function
    def softmax(self,w,x,y,beta,iter):
        # get batch of points
        x_p = x[:,iter]
        y_p = y[:,iter]
        beta_p = beta[:,iter]

        # compute cost over batch        
        cost = np.sum(beta_p*np.log(1 + np.exp(-y_p*self.model(x_p,w))))
        return cost/float(np.size(y_p))

    # the convex relu cost function
    def relu(self,w,x,y,beta,iter):
        # get batch of points
        x_p = x[:,iter]
        y_p = y[:,iter]
        beta_p = beta[:,iter]

        # compute cost over batch
        cost = np.sum(beta_p*np.maximum(0,-y_p*self.model(x_p,w)))
        return cost/float(np.size(y_p))

    ###### multiclass classification costs #######
    # multiclass perceptron
    def multiclass_perceptron(self,w,x,y,beta,iter):
        # get subset of points
        x_p = x[:,iter]
        y_p = y[:,iter]
        beta_p = beta[:,iter]

        # pre-compute predictions on all points
        all_evals = self.model(x_p,w)

        # compute maximum across data points
        a =  np.max(all_evals,axis = 0)        

        # compute cost in compact form using numpy broadcasting
        b = all_evals[y_p.astype(int).flatten(),np.arange(np.size(y_p))]
        cost = np.sum(beta_p*(a - b))

        # return average
        return cost/float(np.size(y_p))

    # multiclass softmax
    def multiclass_softmax(self,w,x,y,beta,iter):     
        # get subset of points
        x_p = x[:,iter]
        y_p = y[:,iter]
        beta_p = beta[:,iter]

        # pre-compute predictions on all points
        all_evals = self.model(x_p,w)
 
        # compute softmax across data points
        a = np.log(np.sum(np.exp(all_evals),axis = 0)) 

        # compute cost in compact form using numpy broadcasting
        b = all_evals[y_p.astype(int).flatten(),np.arange(np.size(y_p))]
        cost = np.sum(beta_p*(a - b))

        # return average
        return cost/float(np.size(y_p))