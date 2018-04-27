# -*- coding: utf-8 -*-

import numpy as np
import cv2
import six
from som32 import SOM32bit
from som8 import SOM8bit

def train32(iters=10000):
    s32 = SOM32bit(16, 16, 3)
    for i in six.moves.range(iters):
        x = np.random.rand(3)
        s32.predict(x, similarity='L2')
        s32.update(0.1, 2.0)
        
        cv2.imshow('map', cv2.resize(s32.w, (200, 200), interpolation=cv2.INTER_NEAREST))
        cv2.waitKey(1)
        
def train8(iters=10000):
    s8 = SOM8bit(16, 16, 3)
    for i in six.moves.range(iters):
        x = (np.random.rand(3) * 255).astype(np.uint8)
        s8.predict(x, similarity='L1')
        s8.update(1)
        
        cv2.imshow('map', cv2.resize(s8.w, (200, 200), interpolation=cv2.INTER_NEAREST))
        cv2.waitKey(1)

train32()     
train8()
