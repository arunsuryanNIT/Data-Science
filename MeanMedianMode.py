# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import statistics as st
import random
import matplotlib.pyplot as mp
lst = []
for i in range(0, 10):
    element = random.randint(1, 10)
    lst.append(element)

print("Mode is ", st.mode(lst), end = " ")
print("Mean is ", st.mean(lst), end = " ")
print("Median is ", st.median(lst), end = " ")

mp.hist(lst, 4)

print(lst)