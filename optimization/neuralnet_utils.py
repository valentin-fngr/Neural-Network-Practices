#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt


# In[ ]:


def sigmoid(Z): 
    '''
        apply the sigmoid function on Z 
        output : 
            s : Z: -> sigmoid(Z)
            cache : a tuple containing the parameter Z
    '''
    
    s = 1 / (1+np.exp(-Z))
    cache = (Z,)
    
    return s, cache 


def linear(A, W, b): 
    '''
        return the linear operation 
        output : 
            s : the linear operation 
            cache : all parameters used for the computation inside a tuple
    '''
    
    assert W.shape[1] == A.shape[0] 
    assert b.shape[0] == W.shape[0]
    
    s = W.dot(A) + b
    cache = (A, W, b) 
    
    return s, cache


def relu(Z): 
    '''
        return the relu operation
        output : 
            s : the relu operation 
            cache : Z inside a tuple
    '''
    
    s = np.maximum(Z, 0) 
    cache = (Z,)
    return s, cache 

def relu_derivative(Z): 
    '''
        return the derivative of the relu function 
        output : 
            A : the derivative of the relu function 
    '''
    A = Z.copy()
    A[A > 0] = 1 
    A[A <= 0] = 0
    
    return A


def compute_cost(A, Y): 
    '''
        return the computation of the cost function 
    '''
    assert A.shape == Y.shape
    m = A.shape[1] 
    return (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))
    
    
def initialize_parameters(X, nb_neurones): 
    '''
        return a dictionary containing all initialized parameters for all layers 
        ouput : 
            the dictionary of parameters
    '''
    parameters = {} 
    
    # get number of samples 
    m = X.shape[1]
    # get number of layers 
    l = len(nb_neurones)
    
    for i in range(l): 
        # init using the He method
        print(f"Initializing for layer : {i}")
        if i == 0: 
            W = np.random.randn(nb_neurones[i], X.shape[0]) * np.sqrt(2/nb_neurones[i])
        else: 
            W = np.random.randn(nb_neurones[i], parameters[f"W{i}"].shape[0]) * np.sqrt(2/nb_neurones[i])
    
        b = np.random.randn(nb_neurones[i], 1) * 0.01
        parameters[f"W{i+1}"] = W 
        parameters[f"b{i+1}"] = b
        
    print("Successfully initialized parameters for the entire network")
        
    return parameters


def forward_pass(X, parameters, Y): 
    '''
        Entire forward propagation across all layers of the network
        output : 
            grads : a dictionnary of all gradients 
            cache : a dictionnary of all parameters used
    '''
    l = len(parameters) // 2 
    grads = {}
    all_cache = {}
    
    A = X
    all_cache["A0"] = A
    for i in range(l): 
        W = parameters[f"W{i+1}"] 
        b = parameters[f"b{i+1}"] 
        
        if i != l-1: 
            # use the relu activation 
            Z, cache = linear(A, W, b)
            
            all_cache[f"W{i+1}"] = W
            all_cache[f"b{i+1}"] = b 
            
            #activation 
            A, cache = relu(Z)
            all_cache[f"Z{i+1}"] = Z 
            all_cache[f"A{i+1}"] = A

        
        else: 
            # last layer has to use the sigmoid activation 
            Z, cache = linear(A, W, b)     
            
            
            all_cache[f"W{i+1}"] = W
            all_cache[f"b{i+1}"] = b 

            # activation 
            A, cache = sigmoid(Z)
            all_cache[f"Z{i+1}"] = Z 
            all_cache[f"A{i+1}"] = A
        
    
    # compute cost
    cost = compute_cost(A, Y) 
    return cost, all_cache

def backward_pass(parameters, cache, Y, l):
    '''
        return all gradients 
        output : 
            grads : a dictionary containing all gradients
    '''
    m = Y.shape[1]
    grads = {} 

    for i in reversed(range(l)): 
        if i == l-1: 
            dZ = cache[f"A{i+1}"] - Y
        else: 
            dZ = cache[f"W{i+2}"].T.dot(dZ) * relu_derivative(cache[f"Z{i+1}"]) 
        
        dW = (1/m) * dZ.dot(cache[f"A{i}"].T)
        db = (1/m) * np.sum(dZ, keepdims=True, axis=1)

        # save gradients 
        grads[f"dZ{i+1}"] = dZ
        grads[f"dW{i+1}"] = dW 
        grads[f"db{i+1}"] = db
            
    return grads
    
def update_parameters(parameters, grads, learning_rate): 
    for param in parameters:
        # takes a long time but do it once ! 
        assert parameters[param].shape == grads[f"d{param}"].shape
        parameters[param] = parameters[param] - learning_rate * grads[f"d{param}"]
    
    return parameters


def predict(X, parameters, Y_true): 
    A = forward_pass(X, parameters, Y_true)[1][f"A{len(parameters) // 2}"]
    pred = A
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    print(f"Accuracy : {(pred == Y_true).mean() * 100} % ")
    return pred

def l_nn(X, Y, nb_neurones, learning_rate, epochs, print_t=False): 
    
    # init parameters 
    parameters = initialize_parameters(X, nb_neurones)
    cost_history = []
    for epoch in range(epochs): 
        # forward pass
        cost, cache = forward_pass(X, parameters, Y) 
        cost_history.append(cost)
        if print_t and epoch % 100 == 0: 
            print(f"Cost value at {epoch} : {cost}")
        # backward propagation 
        grads = backward_pass(parameters, cache, Y, len(nb_neurones))
        prameters = update_parameters(parameters, grads, learning_rate)
        
    print("Training accuracy : ") 
    predictions = predict(X, parameters, Y)
            
    plt.plot(range(epochs), cost_history) 
    plt.show() 
        
    return parameters

