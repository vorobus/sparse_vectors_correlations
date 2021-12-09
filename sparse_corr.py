import numpy as np
import time
class sparse_vector:
    def __init__(self, indexes, values):
        '''
        indexes: where it is non zero
        values: values of it
        '''
        
        self.indexes = np.array(indexes)
        self.values = np.array(values)
        self.shape = self.indexes[-1]
    def __call__(self):
        return self.values
        
    

def correlate_sparse_vector(indices, values,taus,n, time_print = False):
    n = n
    cks = []
    i = indices
    for k in taus:
        ts = time.time()
        i0 = i[i < n-k]   
        ik = (i-k)[i >= k]    
        v0 = values[i < n-k]
        vk = values[i >= k]
        n0 = i0.shape[0]
        nk = ik.shape[0]
        shape_diff = n0 - nk
        if shape_diff > 0:
            ik = np.append(ik, np.arange(ik[-1]+1, ik[-1]+1+shape_diff,1))
            vk = np.append(vk, np.zeros((shape_diff,)))
        elif shape_diff < 0:
            i0 = np.append(i0, np.arange(i0[-1]+1, i0[-1]+1-shape_diff,1))
            v0 = np.append(v0, np.zeros((-shape_diff,)))
        
        t0 = time.time()
        common_elements, kk1,kk2 = np.intersect1d(i0, ik,return_indices=True)
        t1 = time.time()
        #print('common', common_elements)
        #kk1 = [np.where(i0 == ce)[0] for ce in common_elements]
        #kk2 = [np.where(ik == ce)[0] for ce in common_elements]
        t2 = time.time()
        #(a[kk2] == b[kk1]).all()
        #print(kk1, kk2)

        #inds1, = np.where(np.all(i0 == common_elements))
        #inds2, = np.where(np.all(ik == common_elements))
        #print(inds1)
        ck = np.sum(v0[kk1]*vk[kk2])/(n-k)
        cks.append(ck)
        if time_print:
            print(np.array([ts,t0,t1,t2])-ts)
    #plt.plot(k,ck,'r.')
    return cks