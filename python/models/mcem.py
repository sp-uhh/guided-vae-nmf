my_seed = 0
import numpy as np
np.random.seed(my_seed)
import torch
torch.manual_seed(my_seed)
from torch import optim

class EM:
    """
    Meant to be an abstract class that should not be instantiated but only 
    inherited.
    """
    
    def __init__(self, X, W, H, g, vae, niter=100, device='cpu'):
        
        self.device = device
        
        self.X = X.T # mixture STFT, shape (F,N)
        self.X_abs_2 = self.np2tensor(np.abs(X.T)**2).to(self.device) # mixture power spectrogram, shape (F, N)
        self.W = self.np2tensor(W).to(self.device) # NMF dictionary matrix, shape (F, K)
        self.H = self.np2tensor(H).to(self.device) # NMF activation matrix, shape (K, N)
        self.compute_Vb() # noise variance, shape (F, N)
        self.vae = vae # variational autoencoder 
        self.niter = niter # number of iterations
        self.g = g # gain parameters, shape (, N)
        self.Vs = None # speech variance, shape (R, F, N), where R corresponds 
        # to different draws of the latent variables fed as input to the vae
        # decoder
        self.Vs_scaled = None # speech variance multiplied by gain, 
        # shape (R, F, N)
        self.Vx = None # mixture variance, shape (R, F, N)          
    
    def np2tensor(self, x):
        y = torch.from_numpy(x.astype(np.float32))
        return y

    def tensor2np(self, x):
        y = x.numpy()
        return y
    
    def compute_expected_neg_log_like(self):
        #return np.mean(np.log(self.Vx) + self.X_abs_2/self.Vx)
        return torch.mean(torch.log(self.Vx) + self.X_abs_2/self.Vx)
    
    def compute_Vs(self, Z):
        pass        
    
    def compute_Vs_scaled(self):
        self.Vs_scaled = self.g * self.Vs
    
    def compute_Vx(self):
        self.Vx = self.Vs_scaled + self.Vb
        
    def compute_Vb(self):
        self.Vb = self.W @ self.H
    
    def E_step(self):
        # The E-step aims to generate latent variables in order to compute the 
        # cost function required for the M-step. These samples are then also 
        # used to update the model parameters at the M-step.
        pass
    
    def M_step(self):
        # The M-step aims to update W, H and g
        
        if self.Vx.ndim == 2:
            # Vx and Vs are expected to be of shape (R, F, N) so we add a
            # singleton dimension.
            # This will happen for the PEEM algorithm only where there is no
            # sampling of the latent variables according to their posterior.
            rem_dim = True
            self.Vx = self.Vx[np.newaxis,:,:]
            self.Vs = self.Vs[np.newaxis,:,:]
            self.Vs_scaled = self.Vs_scaled[np.newaxis,:,:]
        else:
            rem_dim = False
        
        # update W
        #num = (self.X_abs_2*np.sum(self.Vx**-2, axis=0)) @ self.H.T
        num = (self.X_abs_2*torch.sum(self.Vx**-2, axis=0)) @ self.H.T
        #den = np.sum(self.Vx**-1, axis=0) @ self.H.T
        den = torch.sum(self.Vx**-1, axis=0) @ self.H.T
        self.W = self.W*(num/den)**.5
        
        # update variances
        self.compute_Vb()
        self.compute_Vx()
            
        # update H
        #num = self.W.T @ (self.X_abs_2*np.sum(self.Vx**-2, axis=0))
        num = self.W.T @ (self.X_abs_2*torch.sum(self.Vx**-2, axis=0))
        #den = self.W.T @ np.sum(self.Vx**-1, axis=0)
        den = self.W.T @ torch.sum(self.Vx**-1, axis=0)
        self.H = self.H*(num/den)**.5
        
        # update variances
        self.compute_Vb()
        self.compute_Vx()

        # normalize W and H
        #norm_col_W = np.sum(np.abs(self.W), axis=0)
        norm_col_W = torch.sum(torch.abs(self.W), axis=0)
        #self.W = self.W/norm_col_W[np.newaxis,:]
        self.W = self.W/norm_col_W.unsqueeze(0)
        #self.H = self.H*norm_col_W[:,np.newaxis]
        self.H = self.H*norm_col_W.unsqueeze(1)

        # Update g
        # num = np.sum(self.X_abs_2*np.sum(self.Vs*(self.Vx**-2), axis=0), 
        #              axis=0)
        num = torch.sum(self.X_abs_2*torch.sum(self.Vs*(self.Vx**-2), axis=0), 
                     axis=0)             
        #den = np.sum(np.sum(self.Vs*(self.Vx**-1), axis=0), axis=0)
        den = torch.sum(torch.sum(self.Vs*(self.Vx**-1), axis=0), axis=0)
        self.g = self.g*(num/den)**.5

        # remove singleton dimension if necessary        
        if rem_dim:
            self.Vx = np.squeeze(self.Vx)
            self.Vs = np.squeeze(self.Vs)
            self.Vs_scaled = np.squeeze(self.Vs_scaled)
        
        # update variances
        self.compute_Vs_scaled()
        self.compute_Vx()
        
    
    def run(self):
        
        cost = np.zeros(self.niter)
        
        for n in np.arange(self.niter):
            
            self.E_step()
           
            self.M_step()
            
            cost[n] = self.compute_expected_neg_log_like() # this cost only
            # corresponds to the expectation of the negative log likelihood
            # taken w.r.t the posterior distribution of the latent variables.
            # It basically tells us if the model fits the observations.
            # We could also compute the full variational free energy.
            
            # print("iter %d/%d - cost=%.4f" % (n+1, self.niter, cost[n]))
        
        WFs, WFn = self.compute_WF(sample=True)
            
        self.S_hat = self.tensor2np(WFs.cpu())*self.X
        self.N_hat = self.tensor2np(WFn.cpu())*self.X
            
        return cost


class MCEM_M2(EM):
    
    def __init__(self, X, W, H, g, Z, y, vae, niter, device, nsamples_E_step=10, 
                 burnin_E_step=30, nsamples_WF=25, burnin_WF=75, var_RW=0.01):
        
        super().__init__(X=X, W=W, H=H, g=g, vae=vae, niter=niter, 
             device=device)

        if type(vae).__name__ == 'RVAE':
            raise NameError('MCEM algorithm only valid for FFNN VAE')
        
        self.Z = torch.t(Z) # Last draw of the latent variables, shape (L, N)
        self.y = torch.t(y)  # label
        self.nsamples_E_step = nsamples_E_step
        self.burnin_E_step = burnin_E_step
        self.nsamples_WF = nsamples_WF
        self.burnin_WF = burnin_WF
        self.var_RW = var_RW        
        
        # mixture power spectrogram as tensor, shape (F, N)
        #self.X_abs_2_t = self.np2tensor(self.X_abs_2).to(self.device) 
        self.X_abs_2_t = self.X_abs_2.clone()

        self.vae.eval() # vae in eval mode
        
        
        
    def sample_posterior(self, Z, y, nsamples=10, burnin=30):
        # Metropolis-Hastings
        
        F, N = self.X.shape
        y_dim = self.y.shape[0]
        
        if hasattr(self.vae, 'latent_dim'):
            L = self.vae.latent_dim
        elif hasattr(self.vae, 'z_dim'):
            L = self.vae.z_dim
        
        # random walk variance as tensor
        #var_RM_t = torch.tensor(np.float32(self.var_RW))
        var_RM_t = torch.tensor(np.float32(self.var_RW)).to(self.device)
        
        # latent variables sampled from the posterior
        Z_sampled_t = torch.zeros(N, nsamples, L).to(self.device)
        Z_sampled_y_t = torch.zeros(N, nsamples, L + y_dim).to(self.device) # concat of Z_sampled_t and y
        
        # intial latent variables as tensor, shape (L, N)        
        #Z_t = self.np2tensor(Z).to(self.device)        
        Z_t = Z.clone()
        # speech variances as tensor, shape (F, N)
        #Vs_t = torch.t(self.vae.decoder(torch.t(Z_t)))
        Vs_t = torch.t(self.vae.decoder(torch.t(torch.cat([Z_t, y], dim=0))))
        # vector of gain parameters as tensor, shape (, N)
        #g_t = self.np2tensor(self.g).to(self.device)
        g_t = self.g.clone()
        # noise variances as tensor, shape (F, N)
        #Vb_t = self.np2tensor(self.Vb).to(self.device)
        Vb_t = self.Vb.clone()
        # likelihood variances as tensor, shape (F, N)
        Vx_t = g_t*Vs_t + Vb_t
        
        cpt = 0
        averaged_acc_rate = 0
        for m in np.arange(nsamples+burnin):
            
            # random walk over latent variables
            Z_prime_t = Z_t + torch.sqrt(var_RM_t)*torch.randn(L, N, device=self.device)
            
            # compute associated speech variances
            #Vs_prime_t = torch.t(self.vae.decoder(torch.t(Z_prime_t))) # (F, N)
            Vs_prime_t = torch.t(self.vae.decoder(torch.t(torch.cat([Z_prime_t, y], dim=0))))
            Vs_prime_scaled_t = g_t*Vs_prime_t
            Vx_prime_t = Vs_prime_scaled_t + Vb_t
            
            # compute log of acceptance probability
            acc_prob = ( torch.sum(torch.log(Vx_t) - torch.log(Vx_prime_t) + 
                                   (1/Vx_t - 1/Vx_prime_t)*self.X_abs_2_t, 0) + 
                        .5*torch.sum( Z_t.pow(2) - Z_prime_t.pow(2), 0) )
            
            # accept/reject
            is_acc = torch.log(torch.rand(N, device=self.device)) < acc_prob
            
            # averaged_acc_rate += ( torch.sum(is_acc).numpy()/
            #                       np.prod(is_acc.shape)*100/(nsamples+burnin) )
            
            averaged_acc_rate += ( torch.sum(is_acc).float() /
                                  is_acc.shape.numel()*100/(nsamples+burnin) )

            
            Z_t[:,is_acc] = Z_prime_t[:,is_acc]
            
            # update variances
            #Vs_t = torch.t(self.vae.decoder(torch.t(Z_t)))
            Vs_t = torch.t(self.vae.decoder(torch.t(torch.cat([Z_t, y], dim=0))))
            Vx_t = g_t*Vs_t + Vb_t
            
            if m > burnin - 1:
                Z_sampled_t[:,cpt,:] = torch.t(Z_t)
                Z_sampled_y_t[:,cpt,:] = torch.t(torch.cat([Z_t, y], dim=0))
                cpt += 1
        
        # print('averaged acceptance rate: %f' % (averaged_acc_rate.item()))
        
        return Z_sampled_t, Z_sampled_y_t        
        
            
    def compute_Vs(self, Z):
        """ Z: tensor of shape (N, R, L) """
        with torch.no_grad():  
            Vs_t = self.vae.decoder(Z) # (N, R, F)
        
        if len(Vs_t.shape) == 2: 
            # shape is (N, F) but we need (N, R, F)
            Vs_t = Vs_t.unsqueeze(1) # add a dimension in axis 1
    
        #self.Vs = np.moveaxis(self.tensor2np(Vs_t), 0, -1)  # (R, F, N)        
        self.Vs = Vs_t.unsqueeze(-1).transpose(-1, 0)[0] # Equivalent to np.moveaxis(Vs_t, 0, -1)

    def E_step(self):
        """
        """

        # sample from posterior
        Z_t, Z_y_t = self.sample_posterior(self.Z, self.y, self.nsamples_E_step, 
                                    self.burnin_E_step) # (N, R, L)
        
        # update last draw
        #self.Z = self.tensor2np(torch.squeeze(Z_t[:,-1,:])).T
        self.Z = torch.t(torch.squeeze(Z_t[:,-1,:]))
        
        # compute variances
        #self.compute_Vs(Z_t) 
        self.compute_Vs(Z_y_t) 
        self.compute_Vs_scaled()
        self.compute_Vx()
            
    def compute_WF(self, sample=False):
        
        if sample:
            # sample from posterior
            Z_t, Z_y_t = self.sample_posterior(self.Z, self.y, self.nsamples_WF, 
                                        self.burnin_WF)
            
            # compute variances
            #self.compute_Vs(Z_t)
            self.compute_Vs(Z_y_t) 
            self.compute_Vs_scaled()
            self.compute_Vx()
        
        #WFs = np.mean(self.Vs_scaled/self.Vx, axis=0)
        WFs = torch.mean(self.Vs_scaled/self.Vx, axis=0)
        #WFn = np.mean(self.Vb/self.Vx, axis=0)
        WFn = torch.mean(self.Vb/self.Vx, axis=0)

        return WFs, WFn


class MCEM_M1(EM):
    
    def __init__(self, X, W, H, g, Z, vae, niter, device, nsamples_E_step=10, 
                 burnin_E_step=30, nsamples_WF=25, burnin_WF=75, var_RW=0.01):
        
        super().__init__(X=X, W=W, H=H, g=g, vae=vae, niter=niter, 
             device=device)

        if type(vae).__name__ == 'RVAE':
            raise NameError('MCEM algorithm only valid for FFNN VAE')
        
        self.Z = torch.t(Z) # Last draw of the latent variables, shape (L, N)
        self.nsamples_E_step = nsamples_E_step
        self.burnin_E_step = burnin_E_step
        self.nsamples_WF = nsamples_WF
        self.burnin_WF = burnin_WF
        self.var_RW = var_RW        
        
        # mixture power spectrogram as tensor, shape (F, N)
        #self.X_abs_2_t = self.np2tensor(self.X_abs_2).to(self.device) 
        self.X_abs_2_t = self.X_abs_2.clone()

        self.vae.eval() # vae in eval mode
        
        
        
    def sample_posterior(self, Z, y, nsamples=10, burnin=30):
        # Metropolis-Hastings
        
        F, N = self.X.shape
        
        if hasattr(self.vae, 'latent_dim'):
            L = self.vae.latent_dim
        elif hasattr(self.vae, 'z_dim'):
            L = self.vae.z_dim
        
        # random walk variance as tensor
        #var_RM_t = torch.tensor(np.float32(self.var_RW))
        var_RM_t = torch.tensor(np.float32(self.var_RW)).to(self.device)
        
        # latent variables sampled from the posterior
        Z_sampled_t = torch.zeros(N, nsamples, L).to(self.device)
        #Z_sampled_y_t = torch.zeros(N, nsamples, L + F).to(self.device) # concat of Z_sampled_t and y
        
        # intial latent variables as tensor, shape (L, N)        
        #Z_t = self.np2tensor(Z).to(self.device)        
        Z_t = Z.clone()
        # speech variances as tensor, shape (F, N)
        Vs_t = torch.t(self.vae.decoder(torch.t(Z_t)))
        # vector of gain parameters as tensor, shape (, N)
        #g_t = self.np2tensor(self.g).to(self.device)
        g_t = self.g.clone()
        # noise variances as tensor, shape (F, N)
        #Vb_t = self.np2tensor(self.Vb).to(self.device)
        Vb_t = self.Vb.clone()
        # likelihood variances as tensor, shape (F, N)
        Vx_t = g_t*Vs_t + Vb_t
        
        cpt = 0
        averaged_acc_rate = 0
        for m in np.arange(nsamples+burnin):
            
            # random walk over latent variables
            Z_prime_t = Z_t + torch.sqrt(var_RM_t)*torch.randn(L, N, device=self.device)
            
            # compute associated speech variances
            Vs_prime_t = torch.t(self.vae.decoder(torch.t(Z_prime_t))) # (F, N)
            Vs_prime_scaled_t = g_t*Vs_prime_t
            Vx_prime_t = Vs_prime_scaled_t + Vb_t
            
            # compute log of acceptance probability
            acc_prob = ( torch.sum(torch.log(Vx_t) - torch.log(Vx_prime_t) + 
                                   (1/Vx_t - 1/Vx_prime_t)*self.X_abs_2_t, 0) + 
                        .5*torch.sum( Z_t.pow(2) - Z_prime_t.pow(2), 0) )
            
            # accept/reject
            is_acc = torch.log(torch.rand(N, device=self.device)) < acc_prob
            
            # averaged_acc_rate += ( torch.sum(is_acc).numpy()/
            #                       np.prod(is_acc.shape)*100/(nsamples+burnin) )
            
            averaged_acc_rate += ( torch.sum(is_acc).float() /
                                  is_acc.shape.numel()*100/(nsamples+burnin) )

            
            Z_t[:,is_acc] = Z_prime_t[:,is_acc]
            
            # update variances
            Vs_t = torch.t(self.vae.decoder(torch.t(Z_t)))
            Vx_t = g_t*Vs_t + Vb_t
            
            if m > burnin - 1:
                Z_sampled_t[:,cpt,:] = torch.t(Z_t)
                cpt += 1
        
        print('averaged acceptance rate: %f' % (averaged_acc_rate.item()))
        
        return Z_sampled_t        
        
            
    def compute_Vs(self, Z):
        """ Z: tensor of shape (N, R, L) """
        with torch.no_grad():  
            Vs_t = self.vae.decoder(Z) # (N, R, F)
        
        if len(Vs_t.shape) == 2: 
            # shape is (N, F) but we need (N, R, F)
            Vs_t = Vs_t.unsqueeze(1) # add a dimension in axis 1
    
        #self.Vs = np.moveaxis(self.tensor2np(Vs_t), 0, -1)  # (R, F, N)        
        self.Vs = Vs_t.unsqueeze(-1).transpose(-1, 0)[0] # Equivalent to np.moveaxis(Vs_t, 0, -1)

    def E_step(self):
        """
        """

        # sample from posterior
        Z_t  = self.sample_posterior(self.Z, self.nsamples_E_step, 
                                    self.burnin_E_step) # (N, R, L)
        
        # update last draw
        #self.Z = self.tensor2np(torch.squeeze(Z_t[:,-1,:])).T
        self.Z = torch.t(torch.squeeze(Z_t[:,-1,:]))
        
        # compute variances
        self.compute_Vs(Z_t)  
        self.compute_Vs_scaled()
        self.compute_Vx()
            
    def compute_WF(self, sample=False):
        
        if sample:
            # sample from posterior
            Z_t = self.sample_posterior(self.Z, self.nsamples_WF, 
                                        self.burnin_WF)
            
            # compute variances
            self.compute_Vs(Z_t)
            self.compute_Vs_scaled()
            self.compute_Vx()
        
        #WFs = np.mean(self.Vs_scaled/self.Vx, axis=0)
        WFs = torch.mean(self.Vs_scaled/self.Vx, axis=0)
        #WFn = np.mean(self.Vb/self.Vx, axis=0)
        WFn = torch.mean(self.Vb/self.Vx, axis=0)

        return WFs, WFn
