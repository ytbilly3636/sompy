# -*- coding: utf-8 -*-

import numpy as np
import cv2
import six
from som32 import SOM32bit

s = SOM32bit(8, 8, 3)

for i in six.moves.range(10000):
    x = np.random.rand(3)
    s.predict(x, similarity='L2')
    s.update(0.1, 0.8)
    
    cv2.imshow('map', cv2.resize(s.w, (200, 200), interpolation=cv2.INTER_NEAREST))
    cv2.waitKey(1)
