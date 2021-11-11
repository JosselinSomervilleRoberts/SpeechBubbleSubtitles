# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 01:07:37 2021

@author: josse
"""
from multiprocessing import Pool
import time
import traceback
import sys



def sleep(duration, get_now=time.perf_counter):
    now = get_now()
    end = now + duration
    while now < end:
        now = get_now()
        

def myfunc(x):
    try:
        x = x / 0
        time.sleep(1)
        return x
    except Exception:
        return traceback.format_exc()
 
def mycallback(x):
     print('Callback for i = {}'.format(x))


if __name__ == '__main__':
    pool=Pool()
    
    # Approx of 5s in total
    # Without parallelization, this should take 15s
    t0 = time.time()
    titer = time.time()
    for i in range(100):
        if i% 10 == 0: pool.apply_async(myfunc, (i,), callback=mycallback)
        sleep(0.05) # 50ms
        print("- i =", i, "/ Time iteration:", 1000*(time.time()-titer), "ms")
        titer = time.time()
        
    print("\n\nTotal time:", (time.time()-t0), "s")
    
    t0 = time.time()
    for i in range(100):
        sleep(0.05)
    print("\n\nBenchmark sleep time time:", 10*(time.time()-t0), "ms")
