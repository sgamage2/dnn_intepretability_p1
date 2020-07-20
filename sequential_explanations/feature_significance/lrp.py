import numpy as np


import matplotlib.pyplot as plt
from numpy import newaxis as na

class LRP4LSTM(object):
    def __init__(self, model):
        self.model = model
        
        names = [weight.name for layer in model.layers for weight in layer.weights]
        weights = model.get_weights()

        # suppress scientific notation
        np.set_printoptions(suppress=True)
        for name, weight in zip(names, weights):
            if name == 'lstm_1/kernel:0':
                kernel_0 = weight
            if name == 'lstm_1/recurrent_kernel:0':
                recurrent_kernel_0 = weight
            if name == 'lstm_1/bias:0':
                bias_0 = weight
            elif name == 'dense_1/kernel:0':
                output = weight

        print("kernel_0", kernel_0.shape)
        print("recurrent_kernel_0", recurrent_kernel_0.shape)
        print("bias_0", bias_0.shape)
        print("output", output.shape)

        # self.Wxh_Left (240, 60)
        # self.Whh_Left (240, 60)
        # self.bxh_Left (240,)
        # self.Why_Left (5, 60)

        self.Wxh = kernel_0.T  # shape 4d*e
        self.Whh = recurrent_kernel_0.T  # shape 4d
        self.bxh = bias_0.T  # shape 4d 
        self.Why = output.T
        
    def lrp_linear(self, hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor=1.0, debug=False):
        """
        LRP for a linear layer with input dim D and output dim M.
        Args:
        - hin:            forward pass input, of shape (D,)
        - w:              connection weights, of shape (D, M)
        - b:              biases, of shape (M,)
        - hout:           forward pass output, of shape (M,) (unequal to np.dot(w.T,hin)+b if more than one incoming layer!)
        - Rout:           relevance at layer output, of shape (M,)
        - bias_nb_units:  total number of connected lower-layer units (onto which the bias/stabilizer contribution is redistributed for sanity check)
        - eps:            stabilizer (small positive number)
        - bias_factor:    set to 1.0 to check global relevance conservation, otherwise use 0.0 to ignore bias/stabilizer redistribution (recommended)
        Returns:
        - Rin:            relevance at layer input, of shape (D,)
        """
        sign_out = np.where(hout[na,:]>=0, 1., -1.) # shape (1, M)
        # numerator
        numer    = (w * hin[:,na]) + ( bias_factor * (b[na,:]*1. + eps*sign_out*1.) / bias_nb_units ) # shape (D, M)
        # Note: here we multiply the bias_factor with both the bias b and the stabilizer eps since in fact
        # using the term (b[na,:]*1. + eps*sign_out*1.) / bias_nb_units in the numerator is only useful for sanity check
        # (in the initial paper version we were using (bias_factor*b[na,:]*1. + eps*sign_out*1.) / bias_nb_units instead)
        
        
        # denominator
        denom    = hout[na,:] + (eps*sign_out*1.)   # shape (1, M)

        message  = (numer/denom) * Rout[na,:]       # shape (D, M)

        Rin      = message.sum(axis=1)              # shape (D,)
        
        if debug:
            print("local diff: ", Rout.sum() - Rin.sum())
        # Note: 
        # - local  layer   relevance conservation if bias_factor==1.0 and bias_nb_units==D (i.e. when only one incoming layer)
        # - global network relevance conservation if bias_factor==1.0 and bias_nb_units set accordingly to the total number of lower-layer connections 
        # -> can be used for sanity check

        return Rin
        
    def get_layer_output(self, layer_name, data):
        # 
        intermediate_layer_model = keras.Model(inputs=self.model.input,
                                         outputs=self.model.get_layer(layer_name).output)
        return intermediate_layer_model.predict(data)  
    
    def run(self, target_data, target_class):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
         
        # w_indices [109, 11995, 25, 18263, 25, 973, 3138, 6389, 372]

        x = self.get_layer_output('embedding_1', target_data).squeeze(axis=1)
        e = x.shape[1]

       ################# forward
        T = target_data.shape[0]
        d = int(512/4)  # hidden units
        C = self.Why.shape[0] # number of classes

        idx    = np.hstack((np.arange(0,d), np.arange(2*d,4*d))).astype(int) # indices of gates i,f,o together
        idx_i, idx_f, idx_c, idx_o = np.arange(0,d), np.arange(d,2*d), np.arange(2*d,3*d), np.arange(3*d,4*d) # indices of gates i,g,f,o separately

        
        h  = np.zeros((T,d))
        c  = np.zeros((T,d))
        gates_pre = np.zeros((T, 4*d))  # gates pre-activation
        gates     = np.zeros((T, 4*d))  # gates activation

        for t in range(T):

            gates_pre[t]    = np.dot(self.Wxh, x[t]) + np.dot(self.Whh, h[t-1]) + self.bxh

            gates[t,idx]    = sigmoid(gates_pre[t,idx])
            gates[t,idx_c]  = np.tanh(gates_pre[t,idx_c]) 

            c[t]            = gates[t,idx_f]*c[t-1] + gates[t,idx_i]*gates[t,idx_c]
            h[t]            = gates[t,idx_o]*np.tanh(c[t])

        score = np.dot(self.Why, h[t])    

        ################# backward
        dx     = np.zeros(x.shape)

        dh          = np.zeros((T, d))
        dc          = np.zeros((T, d))
        dgates_pre  = np.zeros((T, 4*d))  # gates pre-activation
        dgates      = np.zeros((T, 4*d))  # gates activation

        ds               = np.zeros((C))
        ds[target_class] = 1.0
        dy               = ds.copy()

       
        dh[T-1]     = np.dot(self.Why.T, dy)
        for t in reversed(range(T)): 
            dgates[t,idx_o]    = dh[t] * np.tanh(c[t])  # do[t]
            dc[t]             += dh[t] * gates[t,idx_o] * (1.-(np.tanh(c[t]))**2) # dc[t]
            dgates[t,idx_f]    = dc[t] * c[t-1]         # df[t]
            dc[t-1]            = dc[t] * gates[t,idx_f] # dc[t-1]
            dgates[t,idx_i]    = dc[t] * gates[t,idx_c] # di[t]
            dgates[t,idx_c]    = dc[t] * gates[t,idx_i] # dg[t]
            dgates_pre[t,idx]  = dgates[t,idx] * gates[t,idx] * (1.0 - gates[t,idx]) # d ifo pre[t]
            dgates_pre[t,idx_c]= dgates[t,idx_c] *  (1.-(gates[t,idx_c])**2) # d c pre[t]
            dh[t-1]            = np.dot(self.Whh.T, dgates_pre[t])
            dx[t]              = np.dot(self.Wxh.T, dgates_pre[t])

        ################# LRP
        eps=0.001 
        bias_factor=1.0
        Rx  = np.zeros(x.shape)
        Rh  = np.zeros((T+1, d))
        Rc  = np.zeros((T+1, d))
        Rg  = np.zeros((T,   d)) # gate g only

        Rout_mask            = np.zeros((C))
        Rout_mask[target_class] = 1.0  

        # format reminder: lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor)
        Rh[T-1]  = self.lrp_linear(h[T-1], self.Why.T, np.zeros((C)), score, score*Rout_mask, d, eps, bias_factor, debug=False)  

        for t in reversed(range(T)):
            Rc[t]   += Rh[t]
            Rc[t-1]  = self.lrp_linear(gates[t,idx_f]*c[t-1], np.identity(d), np.zeros((d)), c[t], Rc[t], d, eps, bias_factor, debug=False)
            Rg[t]    = self.lrp_linear(gates[t,idx_i]*gates[t,idx_c], np.identity(d), np.zeros((d)), c[t], Rc[t], d, eps, bias_factor, debug=False)
            Rx[t]    = self.lrp_linear(x[t], self.Wxh[idx_c].T, self.bxh[idx_c], gates_pre[t,idx_c], Rg[t], d+e, eps, bias_factor, debug=False)
            Rh[t-1]  = self.lrp_linear(h[t-1], self.Whh[idx_c].T, self.bxh[idx_c], gates_pre[t,idx_c], Rg[t], d+e, eps, bias_factor, debug=False)    

        return score, x, dx, Rx, Rh[-1].sum()