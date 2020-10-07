import math
import numpy as np
import torch



class MCEM_M2:
    def __init__(self, X, Z, y, model, device,
                 niter_MCEM=100, niter_MH=40, burnin=30, var_MH=0.01, NMF_rank=8):
        self.X = X
        self.F, self.T = self.X.shape
        eps = np.finfo(float).eps
        self.K = NMF_rank
        np.random.seed(0)
        self.W = np.array(np.maximum(np.random.rand(self.F,self.K), self.eps), dtype='float32')
        self.H = np.array(np.maximum(np.random.rand(self.K,self.T), self.eps), dtype='float32')
        self.V = np.matmul(self.W,self.H) 
        self.Z = Z
        self.y = y
        self.D = self.Z.shape[0]
        self.model = model
        self.g = np.ones((1,self.T), dtype='float32') 
        self.niter_MH = niter_MH 
        self.niter_MCEM = niter_MCEM
        self.burnin = burnin 
        self.var_MH = var_MH 
        self.speech_var = model.decoder(torch.t(torch.cat([self.Z, self.y],axis=0))).cpu().numpy().T * self.g
        self.device = device

    def metropolis_hastings(self, niter_MH, burnin):
        Z_sampled = torch.zeros((self.D, self.T, niter_MH - burnin))
        cpt = 0
        for n in range(niter_MH):
            Z_prime = self.Z + np.sqrt(self.var_MH)*torch.randn(self.D,self.T).to(self.device) 
 
            #S_hat_prime = self.model.decode(Z_prime).cpu().numpy()     
            S_hat_prime = self.model.decoder(torch.t(torch.cat([Z_prime, self.y],axis=0))).cpu().numpy().T
    
            speech_var_prime = S_hat_prime*self.g # apply gain

            acc_prob = (np.sum(np.log(self.V + self.speech_var) - np.log(self.V + speech_var_prime)
                + (1/(self.V + self.speech_var) - 1/(self.V + speech_var_prime))* np.abs(self.X)**2, axis=0)
                + .5*np.sum(self.Z.cpu().numpy()**2 - Z_prime.cpu().numpy()**2, axis=0))
            
            is_acc = np.log(np.random.rand(self.T)) < acc_prob
                        
            self.Z[:,is_acc] = Z_prime[:,is_acc]
            #self.Z_mapped_decoder = self.model.decode(self.Z).cpu().numpy() 
            self.Z_mapped_decoder = self.model.decoder(torch.t(torch.cat([self.Z, self.y],axis=0))).cpu().numpy().T
            self.speech_var = self.Z_mapped_decoder*self.g

            if n > burnin - 1:
                Z_sampled[:,:,cpt] = self.Z
                cpt += 1
        
        return Z_sampled    
    

    def run(self, tol=1e-4):
        X_abs_2 = np.abs(self.X)**2

        cost_after_M_step = np.zeros(self.niter_MCEM)

        for n in range(self.niter_MCEM):
            Z_sampled = self.metropolis_hastings(self.niter_MH, self.burnin)
                      
            Z_sampled_mapped_decoder = []
            for i in range(Z_sampled.shape[2]):
                z_sample = Z_sampled[:,:,i]
                #Z_sample_mapped_decoder = self.model.decode(z_sample.to(self.device)).cpu().numpy()
                Z_sample_mapped_decoder = self.model.decoder(torch.t(torch.cat([z_sample.to(self.device), self.y],axis=0))).cpu().numpy()
                Z_sampled_mapped_decoder.append(Z_sample_mapped_decoder)
            Z_sampled_mapped_decoder = np.stack(Z_sampled_mapped_decoder, axis=2)
            Z_sampled_mapped_decoder = np.moveaxis(Z_sampled_mapped_decoder, 0, 1) # needed to match the size of g
            
            speech_var_multi_samples = Z_sampled_mapped_decoder*self.g[:,:,None] # shape (F,T,N)  
            
            # M-Step
            V_plus_Z_mapped = self.V[:,:,None] + speech_var_multi_samples

            self.W = self.W*(((X_abs_2*np.sum(V_plus_Z_mapped**-2, axis=-1)) @ self.H.T)
                    / (np.sum(V_plus_Z_mapped**-1, axis=-1) @ self.H.T))**.5
            self.V = self.W @ self.H
            V_plus_Z_mapped = self.V[:,:,None] + speech_var_multi_samples

            # Update H
            self.H = self.H*((self.W.T @ (X_abs_2 * np.sum(V_plus_Z_mapped**-2, axis=-1)))
                    / (self.W.T @ np.sum(V_plus_Z_mapped**-1, axis=-1)))**.5
            self.V = self.W @ self.H
            V_plus_Z_mapped = self.V[:,:,None] + speech_var_multi_samples

            # Update gain
            self.g = self.g*((np.sum(X_abs_2 * np.sum(Z_sampled_mapped_decoder*(V_plus_Z_mapped**-2),
                            axis=-1), axis=0))/(np.sum(np.sum(Z_sampled_mapped_decoder*(V_plus_Z_mapped**-1),
                            axis=-1), axis=0)))**.5

            speech_var_multi_samples = (Z_sampled_mapped_decoder*self.g[:,:,None])

            V_plus_Z_mapped = self.V[:,:,None] + speech_var_multi_samples

            cost_after_M_step[n] = np.mean(np.log(V_plus_Z_mapped)+ X_abs_2[:,:,None]/V_plus_Z_mapped )
            
            print('Iteration {}/{}    cost={:4f}'.format(n+1, self.niter_MCEM,
                                        cost_after_M_step[n]), end='\r', file=open('output.log','a'))

            if n>0 and cost_after_M_step[n-1] - cost_after_M_step[n] < tol:
                print('\ntolerance achieved', file=open('output.log','a'))
                break
                
    def separate(self, niter_MH, burnin):
        Z_sampled = self.metropolis_hastings(niter_MH, burnin)
        
        Z_sampled_mapped_decoder = []
        for i in range(Z_sampled.shape[2]):
            z_sample = Z_sampled[:,:,i]
            #Z_sample_mapped_decoder = self.model.decode(z_sample.to(self.device)).cpu().numpy()
            Z_sample_mapped_decoder = self.model.decoder(torch.t(torch.cat([z_sample.to(self.device), self.y],axis=0))).cpu().numpy()
            Z_sampled_mapped_decoder.append(Z_sample_mapped_decoder)
        Z_sampled_mapped_decoder = np.stack(Z_sampled_mapped_decoder, axis=2) 
        Z_sampled_mapped_decoder = np.moveaxis(Z_sampled_mapped_decoder, 0, 1) # needed to match the size of g
               
        speech_var_multi_samples = Z_sampled_mapped_decoder*self.g[:,:,None] # shape (F,T,N)

        self.S_hat = np.mean((speech_var_multi_samples/
                              (speech_var_multi_samples + self.V[:,:,None])), axis=-1) * self.X

        self.N_hat = np.mean((self.V[:,:, None]/
                              (speech_var_multi_samples + self.V[:,:,None])) , axis=-1) * self.X


class MCEM_M1:
    def __init__(self, X, Z, model, device,
                 niter_MCEM=100, niter_MH=40, burnin=30, var_MH=0.01, NMF_rank=8, eps=1e-8):
        self.X = X
        self.F, self.T = self.X.shape
        self.eps = eps
        self.K = NMF_rank
        np.random.seed(0)
        self.W = np.array(np.maximum(np.random.rand(self.F,self.K), self.eps), dtype='float32')
        self.H = np.array(np.maximum(np.random.rand(self.K,self.T), self.eps), dtype='float32')
        self.V = np.matmul(self.W,self.H) 
        self.Z = Z
        self.D = self.Z.shape[0]
        self.model = model
        self.g = np.ones((1,self.T), dtype='float32') 
        self.niter_MH = niter_MH 
        self.niter_MCEM = niter_MCEM
        self.burnin = burnin 
        self.var_MH = var_MH 
        self.speech_var = model.decoder(torch.t(self.Z)).cpu().numpy().T * self.g
        self.device = device

    def metropolis_hastings(self, niter_MH, burnin):
        Z_sampled = torch.zeros((self.D, self.T, niter_MH - burnin))
        cpt = 0
        for n in range(niter_MH):
            #Z_prime = self.Z + torch.tensor(self.var_MH)*torch.randn(self.D,self.T).to(self.device) 
            Z_prime = self.Z + np.sqrt(self.var_MH)*torch.randn(self.D,self.T).to(self.device) 
 
            S_hat_prime = self.model.decoder(torch.t(Z_prime)).cpu().numpy().T
    
            speech_var_prime = S_hat_prime*self.g # apply gain

            acc_prob = (np.sum(np.log(self.V + self.speech_var) - np.log(self.V + speech_var_prime)
                + (1/(self.V + self.speech_var) - 1/(self.V + speech_var_prime))* np.abs(self.X)**2, axis=0)
                + .5*np.sum(self.Z.cpu().numpy()**2 - Z_prime.cpu().numpy()**2, axis=0))
            
            is_acc = np.log(np.random.rand(self.T).astype('float32')) < acc_prob
                        
            self.Z[:,is_acc] = Z_prime[:,is_acc]
            self.Z_mapped_decoder = self.model.decoder(torch.t(self.Z)).cpu().numpy().T
            self.speech_var = self.Z_mapped_decoder*self.g

            if n > burnin - 1:
                Z_sampled[:,:,cpt] = self.Z
                cpt += 1
        
        return Z_sampled    
    

    def run(self, tol=1e-4):
        X_abs_2 = np.abs(self.X)**2

        cost_after_M_step = np.zeros(self.niter_MCEM)

        for n in range(self.niter_MCEM):
            # E-step
            Z_sampled = self.metropolis_hastings(self.niter_MH, self.burnin)
                      
            Z_sampled_mapped_decoder = []
            for i in range(Z_sampled.shape[2]):
                z_sample = Z_sampled[:,:,i]
                Z_sample_mapped_decoder = self.model.decoder(torch.t(z_sample.to(self.device))).cpu().numpy().T
                Z_sampled_mapped_decoder.append(Z_sample_mapped_decoder)
            Z_sampled_mapped_decoder = np.stack(Z_sampled_mapped_decoder, axis=2)
            
            speech_var_multi_samples = Z_sampled_mapped_decoder*self.g[:,:,None] # shape (F,T,N)  
            
            # M-Step
            V_plus_Z_mapped = self.V[:,:,None] + speech_var_multi_samples

            self.W = self.W*(((X_abs_2*np.sum(V_plus_Z_mapped**-2, axis=-1)) @ self.H.T)
                    / (np.sum(V_plus_Z_mapped**-1, axis=-1) @ self.H.T))**.5
            self.V = self.W @ self.H
            V_plus_Z_mapped = self.V[:,:,None] + speech_var_multi_samples

            # Update H
            self.H = self.H*((self.W.T @ (X_abs_2 * np.sum(V_plus_Z_mapped**-2, axis=-1)))
                    / (self.W.T @ np.sum(V_plus_Z_mapped**-1, axis=-1)))**.5
            self.V = self.W @ self.H
            V_plus_Z_mapped = self.V[:,:,None] + speech_var_multi_samples

            # Update gain
            self.g = self.g*((np.sum(X_abs_2 * np.sum(Z_sampled_mapped_decoder*(V_plus_Z_mapped**-2),
                            axis=-1), axis=0))/(np.sum(np.sum(Z_sampled_mapped_decoder*(V_plus_Z_mapped**-1),
                            axis=-1), axis=0)))**.5

            speech_var_multi_samples = (Z_sampled_mapped_decoder*self.g[:,:,None])

            V_plus_Z_mapped = self.V[:,:,None] + speech_var_multi_samples

            cost_after_M_step[n] = np.mean(np.log(V_plus_Z_mapped)+ X_abs_2[:,:,None]/V_plus_Z_mapped )
            
            print('Iteration {}/{}    cost={:4f}'.format(n+1, self.niter_MCEM,
                                        cost_after_M_step[n]), end='\r', file=open('output.log','a'))

            if n>0 and cost_after_M_step[n-1] - cost_after_M_step[n] < tol:
                print('\ntolerance achieved', file=open('output.log','a'))
                break
                
    def separate(self, niter_MH, burnin):
        Z_sampled = self.metropolis_hastings(niter_MH, burnin)
        
        Z_sampled_mapped_decoder = []
        for i in range(Z_sampled.shape[2]):
            z_sample = Z_sampled[:,:,i]
            Z_sample_mapped_decoder = self.model.decoder(torch.t(z_sample.to(self.device))).cpu().numpy().T
            Z_sampled_mapped_decoder.append(Z_sample_mapped_decoder)
        Z_sampled_mapped_decoder = np.stack(Z_sampled_mapped_decoder, axis=2)
               
        speech_var_multi_samples = Z_sampled_mapped_decoder*self.g[:,:,None] # shape (F,T,N)

        self.S_hat = np.mean((speech_var_multi_samples/
                              (speech_var_multi_samples + self.V[:,:,None])), axis=-1) * self.X

        self.N_hat = np.mean((self.V[:,:, None]/
                              (speech_var_multi_samples + self.V[:,:,None])) , axis=-1) * self.X