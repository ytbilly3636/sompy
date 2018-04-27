# -*- coding: utf-8 -*-

import numpy as np

class SOM8bit:
    def __init__(self, height_map, width_map, size_vector):
        '''
        height_map:     grid height of neurons
        width_map:      grid width of neurons
        size_vector:    dimension of input vector 
        '''
        
        # matrix of weight
        self.w = np.zeros((height_map, width_map, size_vector), np.uint8)


    def __manhattan_distance__(self, cy, cx):
        '''
        return gaussian function
        
        cy:             index of height
        cx:             index of width
        '''
        
        # using meshgrid, generate grid of distance from winner
        x_axis = np.abs(np.arange(-cx, self.w.shape[1] - cx))
        y_axis = np.abs(np.arange(-cy, self.w.shape[0] - cy))
        xs, ys = np.meshgrid(x_axis, y_axis)
        manhattan_distance = xs + ys
        manhattan_distance = manhattan_distance.reshape(self.w.shape[0], self.w.shape[1], 1)
        return manhattan_distance
    
    
    def __arith_right_bitshift__(self, ary, bit):
        '''
        arithmetic bitshift operation (considering sign)
        
        ary:            input array
        bit:            amount of bit shift
        '''
        
        # store sign of negative elements
        sign = np.ones(ary.shape)
        sign[np.where(ary < 0)] = -1.0
        
        # right bitshift of absolute value
        shifted = np.abs(ary) >> bit
        
        # deconvert
        shifted = shifted * sign
        return shifted
        
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
        
        
    def update(self, lr_shift):
        '''
        update weight vector
        run this function after predict
        
        lr_shift:       amount of bit shift (corresponding to learning rate)  
        '''
        
        # neighbourhood function
        h = self.__manhattan_distance__(self.win[0], self.win[1])
        dw = self.x.astype(np.int32) - self.w.astype(np.int32)
        dw = self.__arith_right_bitshift__(dw, lr_shift + h)
        w = self.w + dw
        w[np.where(w > 255)] = 255
        w[np.where(w < 0)] = 0
        self.w = w.astype(np.uint8)
