'''

'''

import numpy as np
import torch
from matplotlib import pyplot as plt

class Transformer:
    def __init__(self,I,J, Ires, Jres,
                 nt=5,a=2.0,p=2.0,
                 sigmaM=1.0,sigmaR=1.0,
                 order=2,
                 sigmaA=None,
                 transformer=None, 
                 A=None, v=None, device=None):
        '''
        Specify polynomial intensity mapping order with order parameters
        2 corresponds to linear, nothing less than 2 is supported
        input sigmaA for weights
        
        assume input images are and gridpoints are torch tensors already

        If transformer is not None (assumed to be a Transformer instance), 
        its A and v attributes are used, unless they are provided as arguments.
        '''
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda:0'
            else:
                self.device = 'cpu'
        elif 'cuda' in device or device == 'cpu':
            self.device = device
        else:
            raise ValueError(f"inappropriate value for device. device: {device}.")
        self.dtype = torch.float64
        
        self.I = torch.tensor(I, dtype=self.dtype, device=self.device)
        self.J = torch.tensor(J, dtype=self.dtype, device=self.device)
        self.Ires = Ires
        self.Jres = Jres
        
        # self.I = torch.tensor(I, dtype=self.dtype, device=self.device)
        xI = [np.arange(nxyz_i)*dxyz_i - np.mean(np.arange(nxyz_i)*dxyz_i) for nxyz_i, dxyz_i in zip(I.shape, Ires)] # Create coords as a list of numpy arrays.
        xI = [torch.tensor(xI_i, dtype=self.dtype, device=self.device) for xI_i in xI] # Convert to lists of tensors.
        self.xI = xI
        self.nxI = I.shape
        self.dxI = torch.tensor([xI[0][1]-xI[0][0], xI[1][1]-xI[1][0], xI[2][1]-xI[2][0]],
                                dtype=self.dtype,device=self.device)
        self.XI = torch.stack(torch.meshgrid(xI))
        
        # self.J = torch.tensor(J, dtype=self.dtype, device=self.device)
        xJ = [np.arange(nxyz_i)*dxyz_i - np.mean(np.arange(nxyz_i)*dxyz_i) for nxyz_i, dxyz_i in zip(J.shape, Jres)] # Create coords as a list of numpy arrays.
        xJ = [torch.tensor(xJ_i, dtype=self.dtype, device=self.device) for xJ_i in xJ] # Convert to lists of tensors.
        self.xJ = xJ
        self.nxJ = J.shape
        self.dxJ = torch.tensor([xJ[0][1]-xJ[0][0], xJ[1][1]-xJ[1][0], xJ[2][1]-xJ[2][0]],
                                dtype=self.dtype,device=self.device)
        self.XJ = torch.stack(torch.meshgrid(xJ))
        
        # a weight, may be updated via EM
        self.WM = torch.ones(self.nxJ,dtype=self.dtype,device=self.device)
        if sigmaA is not None:
            self.WM *= 0.9
            self.WA = torch.ones(self.nxJ,dtype=self.dtype,device=self.device)*0.1
            self.CA = torch.max(J) # constant value for artifact
        
        self.nt = nt
        self.dt = 1.0/nt
        
        self.sigmaM = sigmaM
        self.sigmaR = sigmaR
        self.sigmaA = sigmaA
        
        self.order = order
        
        self.EMsave = []
        self.ERsave = []
        self.Esave = []
        
        usegrad = False # typically way too much memory

        if v is not None:
            self.v = torch.tensor(v, dtype=self.dtype, device=self.device)
        elif transformer is not None:
            if hasattr(transformer, 'v'):
                self.v = transformer.v
            else:
                # TODO: fix redundant code.
                self.v = torch.zeros((self.nt,3,self.nxI[0],self.nxI[1],self.nxI[2]),
                        dtype=self.dtype,device=self.device, requires_grad=usegrad)
        else:
            self.v = torch.zeros((self.nt,3,self.nxI[0],self.nxI[1],self.nxI[2]),
                        dtype=self.dtype,device=self.device, requires_grad=usegrad)
        self.vhat = torch.rfft(self.v,3,onesided=False)
        
        if A is not None:
            self.A = torch.tensor(A, dtype=self.dtype, device=self.device)
        elif transformer is not None:
            if hasattr(transformer, 'A'):
                self.A = transformer.A
            else:
                # TODO: fix redundant code.
                self.A = torch.eye(4,dtype=self.dtype,device=self.device, requires_grad=usegrad)
        else:
            self.A = torch.eye(4,dtype=self.dtype,device=self.device, requires_grad=usegrad)

        # smoothing
        f0I = torch.arange(self.nxI[0],dtype=self.dtype,device=self.device)/self.dxI[0]/self.nxI[0]
        f1I = torch.arange(self.nxI[1],dtype=self.dtype,device=self.device)/self.dxI[1]/self.nxI[1]
        f2I = torch.arange(self.nxI[2],dtype=self.dtype,device=self.device)/self.dxI[2]/self.nxI[2]
        F0I,F1I,F2I = torch.meshgrid(f0I, f1I, f2I)
        self.a = a
        self.p = p # fourier_high_pass_filter_power
        Lhat = (1.0 - self.a**2*( (-2.0 + 2.0*torch.cos(2.0*np.pi*self.dxI[0]*F0I))/self.dxI[0]**2 
                + (-2.0 + 2.0*torch.cos(2.0*np.pi*self.dxI[1]*F1I))/self.dxI[1]**2
                + (-2.0 + 2.0*torch.cos(2.0*np.pi*self.dxI[2]*F2I))/self.dxI[2]**2 ) )**self.p
        self.Lhat = Lhat
        self.LLhat = self.Lhat**2
        self.Khat = 1.0/self.LLhat
        
    def forward(self):        
        ################################################################################
        # flow forwards
        self.phii = self.XI.clone().detach() # recommended way to copy construct from  a tensor
        self.It = torch.zeros((self.nt,self.nxI[0],self.nxI[1],self.nxI[2]),dtype=self.dtype,device=self.device)
        self.It[0] = self.I
        for t in range(self.nt):
            # apply the tform to I0    
            if t > 0: self.It[t] = self.interp3(self.xI,self.I,self.phii)
            Xs = self.XI - self.dt*self.v[t]
            self.phii = self.interp3(self.xI,self.phii-self.XI,Xs) + Xs
        # apply deformation including affine
        self.Ai = torch.inverse(self.A)
        X0s = self.Ai[0,0]*self.XJ[0] + self.Ai[0,1]*self.XJ[1] + self.Ai[0,2]*self.XJ[2] + self.Ai[0,3]
        X1s = self.Ai[1,0]*self.XJ[0] + self.Ai[1,1]*self.XJ[1] + self.Ai[1,2]*self.XJ[2] + self.Ai[1,3]
        X2s = self.Ai[2,0]*self.XJ[0] + self.Ai[2,1]*self.XJ[1] + self.Ai[2,2]*self.XJ[2] + self.Ai[2,3]
        self.AiX = torch.stack([X0s,X1s,X2s])
        self.phiiAi = self.interp3(self.xI,self.phii-self.XI,self.AiX) + self.AiX
        self.AphiI = self.interp3(self.xI,self.I,self.phiiAi)
        ################################################################################
        # calculate and apply intensity transform        
        AphiIflat = torch.flatten(self.AphiI)
        Jflat = torch.flatten(self.J)
        WMflat = torch.flatten(self.WM)
        # format data into a Nxorder matrix
        B = torch.zeros((self.AphiI.numel(),self.order), device=self.device, dtype=self.dtype) # B for basis functions
        for o in range(self.order):
            B[:,o] = AphiIflat**o
        BT = torch.transpose(B,0,1)
        BTB = torch.matmul( BT*WMflat, B)        
        BTJ = torch.matmul( BT*WMflat, Jflat )              
        self.coeffs,_ = torch.solve(BTJ[:,None],BTB) 
        self.CA = torch.mean(self.J*(1.0-self.WM))
        # torch.solve(B,A) solves AX=B (note order is opposite what I'd expect)        
        self.fAphiI = torch.matmul(B,self.coeffs).reshape(self.nxJ)
        # for convenience set this error to a member
        self.err = self.fAphiI - self.J
        
    def weights(self):
        '''Calculate image matching and artifact weights in simple Gaussian mixture model'''
        fM = torch.exp( (self.fAphiI - self.J)**2*(-1.0/2.0/self.sigmaM**2) ) / np.sqrt(2.0*np.pi*self.sigmaM**2)
        fA = torch.exp( (self.CA     - self.J)**2*(-1.0/2.0/self.sigmaA**2) ) / np.sqrt(2.0*np.pi*self.sigmaA**2)
        fsum = fM + fA
        self.WM = fM/fsum        
        
    def cost(self):                
        # get matching cost
        EM = torch.sum((self.fAphiI - self.J)**2*self.WM)/2.0/self.sigmaM**2*torch.prod(self.dxJ)
        # note the complex number is just an extra dimension at the end
        # note divide by numel(I) to conserve power when summing in fourier domain
        ER = torch.sum(torch.sum(torch.sum(torch.abs(self.vhat)**2,dim=(-1,1,0))*self.LLhat))\
            *(self.dt*torch.prod(self.dxI)/2.0/self.sigmaR**2/torch.numel(self.I))        
        E = ER + EM     
        # append these outputs for plotting
        self.EMsave.append(EM.cpu().numpy())
        self.ERsave.append(ER.cpu().numpy())
        self.Esave.append(E.cpu().numpy())
        
    def step_v(self, eV=0.0):
        ''' One step of gradient descent for velocity field v'''
        # get error
        err = (self.fAphiI - self.J)*self.WM # derivative of matching wrt it's arg
        # propagate error through poly
        Df = torch.zeros(self.nxJ, device=self.device, dtype=self.dtype)            
        for o in range(1,self.order):
            Df +=  o * self.AphiI**(o-1) *self.coeffs[o]
        errDf = err * Df # derivative of matching wrt the transformed image
        # deform back through flow
        self.phi = self.XI.clone().detach() # torch recommended way to make a copy
        for t in range(self.nt-1,-1,-1):
            Xs = self.XI + self.dt*self.v[t]
            self.phi = self.interp3(self.xI,self.phi-self.XI,Xs) + Xs
            Aphi0 = self.A[0,0]*self.phi[0] + self.A[0,1]*self.phi[1] + self.A[0,2]*self.phi[2] + self.A[0,3]
            Aphi1 = self.A[1,0]*self.phi[0] + self.A[1,1]*self.phi[1] + self.A[1,2]*self.phi[2] + self.A[1,3]
            Aphi2 = self.A[2,0]*self.phi[0] + self.A[2,1]*self.phi[1] + self.A[2,2]*self.phi[2] + self.A[2,3]
            self.Aphi = torch.stack([Aphi0,Aphi1,Aphi2])
            # gradient
            Dphi = self.gradient(self.phi,self.dxI)
            detDphi = Dphi[0][0]*(Dphi[1][1]*Dphi[2][2]-Dphi[1][2]*Dphi[2][1]) \
                - Dphi[0][1]*(Dphi[1][0]*Dphi[2][2] - Dphi[1][2]*Dphi[2][0]) \
                + Dphi[0][2]*(Dphi[1][0]*Dphi[2][1] - Dphi[1][1]*Dphi[2][0])
            # pull back error
            errDft = self.interp3(self.xJ,errDf,self.Aphi)
            
            # gradient of image
            DI = self.gradient(self.It[t],self.dxI)
            # the gradient, error, times, determinant, times image grad
            grad = (errDft*detDphi)[None]*DI*(-1.0/self.sigmaM**2)*torch.det(self.A)
            # smooth it (add extra dimension for complex)
            gradhats = torch.rfft(grad,3,onesided=False)*self.Khat[...,None]
            # add reg
            gradhats = gradhats + self.vhat[t]/self.sigmaR**2
            # get final gradient
            grad = torch.irfft(gradhats,3,onesided=False)
            # update
            self.v[t] -= grad*eV
        # fourier transform for later
        self.vhat = torch.rfft(self.v,3,onesided=False)
           
    def step_A(self,eL=0.0,eT=0.0):        
        # get error
        err = (self.fAphiI - self.J)*self.WM
        # energy gradient with respect to affine transform
        DfAphiI = self.gradient(self.fAphiI,dx=self.dxJ)
        DfAphiI0 = torch.cat((DfAphiI,torch.zeros(self.nxJ,dtype=self.dtype,device=self.device)[None]))
        # gradient should go down a row, X across a column
        AiXo = torch.cat((self.AiX,torch.ones(self.nxJ,dtype=self.dtype,device=self.device)[None]))
        gradA = torch.sum(DfAphiI0[:,None,...]*AiXo[None,:,...]*err[None,None],(-1,-2,-3))\
            *(-1.0/self.sigmaM**2*torch.prod(self.dxI))            
        gradA = torch.matmul(torch.matmul(self.Ai.t(),gradA),self.Ai.t())
        
        # update A
        EL = torch.tensor([[1,1,1,0],[1,1,1,0],[1,1,1,0],[0,0,0,0]],dtype=self.dtype,device=self.device)
        ET = torch.tensor([[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,0]],dtype=self.dtype,device=self.device)
        e = EL*eL + ET*eT            
        stepA = e*gradA       
        self.A = self.A - stepA
        self.Ai = torch.inverse(self.A)
        
    # to interpolate, use this
    # https://pytorch.org/docs/0.3.0/nn.html#torch.nn.functional.grid_sample
    def interp3(self,x,I,phii):
        '''Interpolate image I,
        sampled at points x (1d array), 
        at the new points phii (dense grid)     
        Note that
        grid[n, d, h, w] specifies the x, y, z pixel locations 
        for interpolating output[n, :, d, h, w]
        '''
        # unpack arguments
        x0 = x[0]
        x1 = x[1]
        x2 = x[2]
        phii0 = phii[0]
        phii1 = phii[1]
        phii2 = phii[2]
        # the input image needs to be reshaped so the first two dimensions are 1
        if I.dim() == 3:
            # for grayscale images
            Ireshape = I[None,None,...]
        elif I.dim() == 4:
            # for vector fields, multi channel images, etc.
            Ireshape = I[None,...]
        else:
            raise ValueError('Tensor to interpolate must be dim 3 or 4')
        # the coordinates need to be rescaled from -1 to 1    
        grid0 = (phii0 - x0[0])/(x0[-1] - x0[0])*2.0 - 1.0
        grid1 = (phii1 - x1[0])/(x1[-1] - x1[0])*2.0 - 1.0
        grid2 = (phii2 - x2[0])/(x2[-1] - x2[0])*2.0 - 1.0
        grid = torch.stack([grid2,grid1,grid0], dim=-1)
        # and the grid also needs to be reshaped to have a 1 as the first index
        grid = grid[None]
        # do the resampling
        out = torch.nn.functional.grid_sample(Ireshape, grid, padding_mode='border')
        # squeeze out the first dimensions
        if I.dim()==3:
            out = out[0,0,...]
        elif I.dim() == 4:
            out = out[0,...]
        # return the output
        return out
    
    # now we need gradient
    def gradient(self,I,dx=[1,1,1]):
        ''' Gradient of an image in each direction
        We want centered difference in the middle, 
        and forward or backward difference at the ends
        image I can have as many leading dimensions as you want, 
        gradient will apply to the last three
        '''
        I_0_list = [ (I[...,1,:,:]-I[...,0,:,:])[...,None,:,:]/dx[0],
                (I[...,2:,:,:]-I[...,:-2,:,:])/(2.0*dx[0]),
                (I[...,-1,:,:]-I[...,-2,:,:])[...,None,:,:]/dx[0] ]
        I_0 = torch.cat(I_0_list, dim=-3)

        I_1_list = [ (I[...,:,1,:]-I[...,:,0,:])[...,:,None,:]/dx[1],
            (I[...,:,2:,:]-I[...,:,:-2,:])/(2.0*dx[1]),
            (I[...,:,-1,:]-I[...,:,-2,:])[...,:,None,:]/dx[1] ]
        I_1 = torch.cat(I_1_list, dim=-2)

        I_2_list = [ (I[...,:,:,1]-I[...,:,:,0])[...,:,:,None]/dx[2],
            (I[...,:,:,2:]-I[...,:,:,:-2])/(2.0*dx[2]),
            (I[...,:,:,-1]-I[...,:,:,-2])[...,:,:,None]/dx[2] ]
        I_2 = torch.cat(I_2_list, dim=-1)
        # note we insert the new dimension at position -4!
        return torch.stack([I_0,I_1,I_2],-4)
    
    def show_image(self,I,x=None,n=None,fig=None,clim=None):
        if n is None:
            n = 6
        if x is None:        
            x = [np.arange(n) - np.mean(np.arange(n)) for n in I.shape]
        slices = np.linspace(0,I.shape[2],n+2)
        slices = np.round(slices[1:-1]).astype(int)
        if fig is None:
            fig,ax = plt.subplots(1,n)
        else:
            fig.clf()
            ax = fig.subplots(1,n)

        if clim is None:
            clim = [np.min(I),np.max(I)]
            m = np.median(I)
            sad = np.mean(np.abs(I - m))
            nsad = 4.0
            clim = [m-sad*nsad, m+sad*nsad]

        for s in range(n):        
            ax[s].imshow(I[:,:,slices[s]], 
                         extent=(x[1][0],x[1][-1],x[0][0],x[0][-1]), 
                         origin='lower',vmin=clim[0],vmax=clim[1],cmap='gray')
            if s > 0:
                ax[s].set_yticklabels([])
            ax[s].set_title(f'z={slices[s]}')
        return fig,ax


'''torch_register and torch_apply'''


def torch_register(template, target, transformer, sigmaR, eV, eL=0, eT=0, **kwargs):
    """daniel's version for demo to be replaced
    Perform a registration between <template> and <target>.
    Supported kwargs [default value]:
    a [2]-> smoothing kernel, in # of voxels
    niter -> total iteration limit
    eT [0] -> translation step size
    eL [0] -> linear transformation step size
    eV -> deformative step size
    sigmaR -> deformation allowance
    do_affine [0]-> enable affine transformation (0 or 1)
    outdir -> ['.'] output directory path
   """
    # Set defaults.
    arguments = {
        'a':2, # smoothing kernel, scaled to pixel size
        'p':2,
        'niter':200,
        'naffine':50, 
        'eV':eV, # velocity
        'eL':eL, # linear
        'eT':eT, # translation
        'sigmaM':1.0, # sigmaM
        'sigmaR':sigmaR, # sigmaR
        'sigmaA':None, # for EM algorithm
        'nt':3, # number of time steps in velocity field           
        'order':2, # polynomial order
        'draw':False,
        'tune':False,
    }
    # Update parameters with kwargs.
    arguments.update(kwargs)
    
    device = transformer.device
    dtype = transformer.dtype
    
    if arguments['draw']:
        plt.ion()
        f1 = plt.figure()
        f2 = plt.figure()
        if arguments['sigmaA'] is not None:
            f3 = plt.figure()
    vmaxsave = [] # for visualization, maximum velocity
    Lsave = [] # for visualization, linear transform
    Tsave = [] # for visualizatoin, translation
    for it in range(arguments['niter']):
        transformer.forward()
        transformer.cost()
        if arguments['sigmaA'] is not None:
            transformer.weights()
        if it >= arguments['naffine'] and arguments['eV']>-1.0:
            transformer.step_v(eV=arguments['eV'])
        transformer.step_A(eT=arguments['eT'],eL=arguments['eL'])
        
        if arguments['draw'] and not it%5:        
            plt.close(f1)
            f1 = plt.figure()
            ax = f1.add_subplot(1,1,1)    
            ERsave = transformer.ERsave    
            EMsave = transformer.EMsave
            Esave = transformer.Esave
            ax.plot(ERsave)
            ax.plot(EMsave)
            ax.plot(Esave)
            ax.legend(['ER','EM','E'])
            f1.canvas.draw()
            
            plt.close(f2)
            f2 = plt.figure()
            #transformer.show_image(transformer.fAphiI.cpu().numpy(),fig=f2) # or show err?
            transformer.show_image(transformer.err.cpu().numpy(),fig=f2) # or show err?
            f2.canvas.draw()
            
            if arguments['sigmaA'] is not None:
                plt.close(f3)
                f3 = plt.figure()
                transformer.show_image(transformer.WM.cpu().numpy(),fig=f3,clim=[0,1])            
                f3.canvas.draw()
                
            plt.pause(0.0001)
            
        vmax = (torch.max(torch.sum(transformer.v.detach()**2, dim=1))**0.5).cpu().numpy()
        vmaxsave.append(vmax)
        L = transformer.A[:3,:3].detach().cpu().numpy()
        Lsave.append(L)
        T = transformer.A[:3,-1].detach().cpu().numpy()
        Tsave.append(T)
        if not it % 10:
            print(f'Completed iteration {it}, E={transformer.Esave[-1]}, EM={transformer.EMsave[-1]}, ER={transformer.ERsave[-1]}')
        
    # Display final images.
    if arguments['tune']:
        f, axs = plt.subplots(2,2)
        axs[0,0].plot(list(zip(transformer.Esave,transformer.EMsave,transformer.ERsave)))
        xlim = axs[0,0].get_xlim()
        ylim = axs[0,0].get_ylim()
        axs[0,0].set_aspect((xlim[1]-xlim[0])/(ylim[1]-ylim[0]))
        axs[0,0].legend(['Etot','Ematch','Ereg'])
        axs[0,0].set_title('Energy minimization')
        
        axs[0,1].plot(Tsave)
        axs[0,1].set_title('Translation')
        axs[0,1].legend(['x0','x1','x2'])
        
        axs[1,0].plot( np.array(Lsave).reshape((-1,9)) )
        axs[1,0].set_title('Linear')
        
        
        axs[1,1].plot(vmaxsave)
        axs[1,1].set_title('Maximum velocity')
        
        
    
    return {
        'Aphis':transformer.Aphi.cpu().numpy() if hasattr(transformer, 'Aphi') else None, 
        'phis':transformer.phi.cpu().numpy() if hasattr(transformer, 'Aphi') else None, 
        'phiinvs':transformer.phii.cpu().numpy(), 
        'phiinvAinvs':transformer.phiiAi.cpu().numpy(), 
        'A':transformer.A.cpu().numpy(), 
        'transformer':transformer, 
        }


def torch_apply_transform(image:np.ndarray, deform_to='template', transformer=None):
    """daniel's version for demo to be replaced
    Apply the transformation stored in Aphis (for deforming to the template) and phiinvAinvs (for deforming to the target).
    If deform_to='template', Aphis must be provided.
    If deform_to='target', phiinvAinvs must be provided."""
    # Presently must be given transformer.
    if transformer is None:
        raise RuntimeError("transformer must be provided with present implementation.")

    if deform_to == 'template':
        out = transformer.interp3(transformer.xJ,torch.tensor(image,dtype=transformer.dtype,device=transformer.device),transformer.Aphi)
    elif deform_to == 'target':
        out = transformer.interp3(transformer.xI,torch.tensor(image,dtype=transformer.dtype,device=transformer.device),transformer.phiiAi)
    elif deform_to == 'template-identity': # deform to template with identity
        out = transformer.interp3(transformer.xJ,torch.tensor(image,dtype=transformer.dtype,device=transformer.device),transformer.XI)
    elif deform_to == 'target-identity':
        out = transformer.interp3(transformer.xI,torch.tensor(image,dtype=transformer.dtype,device=transformer.device),transformer.XJ)
    return out.cpu().numpy()