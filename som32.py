# -*- coding: utf-8 -*-

import numpy as np

class SOM32bit:
    def __init__(self, height_map, width_map, size_vector):
        '''
        height_map:     grid height of neurons
        width_map:      grid width of neurons
        size_vector:    dimension of input vector 
        '''
        
        # matrix of weight
        self.w = np.zeros((height_map, width_map, size_vector), np.float32)


    def __gaussian__(self, cy, cx, var):
        '''
        return gaussian function
        
        cy:             index of height
        cx:             index of width
        var:            variance of gaussian
        '''
        
        # using meshgrid, generate grid of distance from winner
        x_axis = np.abs(np.arange(-cx, self.w.shape[1] - cx))
        y_axis = np.abs(np.arange(-cy, self.w.shape[0] - cy))
        xs, ys = np.meshgrid(x_axis, y_axis)
        distance = (xs ** 2 + ys ** 2)
        
        # gaussian function
        gauss = np.exp(-distance / (var ** 2 * 2))
        gauss = gauss.reshape(self.w.shape[0], self.w.shape[1], 1)
        return gauss
        
        
    def predict(self, x, similarity='L2'):
        '''
        searching winner neuron
        return indices of winner
        
        x:              input vector
        similarity:     'L1' or 'L2', similarity between input (x) and weight (w)
        '''
        
        # store input vector
        self.x = x.reshape(1, 1, -1)
        
        # winner has minimum L1 or L2 norm between input and weight
        sim = np.sum(np.abs(x - self.w), axis=2) if similarity == 'L1' else np.sum((x - self.w) ** 2, axis=2)
        win = np.argmin(sim)
        self.win = win / self.w.shape[1], win % self.w.shape[1]
        return self.win
        
        
    def update(self, lr, var):
        '''
        update weight vector
        run this function after predict
        
        lr:             learning rate
        var:            update range          
        '''
        
        # neighbourhood function
        h = self.__gaussian__(self.win[0], self.win[1], var)
        self.w += (self.x - self.w) * lr * h
