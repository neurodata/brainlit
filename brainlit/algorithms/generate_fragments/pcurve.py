#Copyright 2020, zsteve
#Written entirely by https://github.com/zsteve
#https://github.com/zsteve/pcurvepy
#This code was slightly modified by https://github.com/CaseyWeiner, 2022
#CaseyWeiner added an "s_factor" parameter to init to allow smoothing factor
#customization in the univariate spline interpolation step.

import sklearn
import numpy as np
from sklearn.decomposition import PCA
from scipy.interpolate import UnivariateSpline

class PrincipalCurve:
    def __init__(self, k = 3, s_factor=1):
        self.k = k
        self.p = None
        self.s = None
        self.p_interp = None
        self.s_interp = None
        self.s_factor = s_factor
        
    def project(self, X, p, s):
        """Get interpolating s values for projection of X onto the curve defined by (p, s)

        Args:
            X (np.ndarray): data
            p (np.ndarray): curve points
            s (np.ndarray): curve parameterisation

        Returns:
            np.ndarray: interpolating parameter values
            np.ndarray: projected points on curve
            np.ndarray: sum of square distances
        """
        s_interp = np.zeros(X.shape[0])
        p_interp = np.zeros(X.shape)
        d_sq = 0
        
        for i in range(0, X.shape[0]):
            z = X[i, :]
            seg_proj = (((p[1:] - p[0:-1]).T)*np.einsum('ij,ij->i', z - p[0:-1], p[1:] - p[0:-1])/np.power(np.linalg.norm(p[1:] - p[0:-1], axis = 1), 2)).T # compute parallel component
            proj_dist = (z - p[0:-1]) - seg_proj # compute perpendicular component 
            dist_endpts = np.minimum(np.linalg.norm(z - p[0:-1], axis = 1), np.linalg.norm(z - p[1:], axis = 1))
            dist_seg = np.maximum(np.linalg.norm(proj_dist, axis = 1), dist_endpts)

            idx_min = np.argmin(dist_seg)
            q = seg_proj[idx_min] 
            s_interp[i] = (np.linalg.norm(q)/np.linalg.norm(p[idx_min + 1, :] - p[idx_min, :]))*(s[idx_min+1]-s[idx_min]) + s[idx_min]
            p_interp[i] = (s_interp[i] - s[idx_min])*(p[idx_min+1, :] - p[idx_min, :]) + p[idx_min, :]
            d_sq = d_sq + np.linalg.norm(proj_dist[idx_min])**2
            
        return s_interp, p_interp, d_sq
     
    def renorm_parameterisation(self, p):
        """Renormalise curve to unit speed

        Args:
            p (np.ndarray): curve points

        Returns:
            np.ndarray: new parameterisation
        """
        seg_lens = np.linalg.norm(p[1:] - p[0:-1], axis = 1)
        s = np.zeros(p.shape[0])
        s[1:] = np.cumsum(seg_lens)
        s = s/sum(seg_lens)
        return s
    
    def fit(self, X, p = None, w = None, max_iter = 10, tol = 1e-3):
        """Fit principal curve to data

        Args:
            X (np.ndarray): data
            p (np.ndarray): starting curve (optional)
            w (np.ndarray): data weights (optional)
            max_iter (int): maximum number of iterations 
            tol (float): tolerance for stopping condition
        """
        pca = sklearn.decomposition.PCA(n_components = X.shape[1])
        pca.fit(X)
        pc1 = pca.components_[:, 0]
        if p is None:
            p = np.kron(np.dot(X, pc1)/np.dot(pc1, pc1), pc1).reshape(X.shape) # starting point for iteration
            order = np.argsort([np.linalg.norm(p[0, :] - p[i, :]) for i in range(0, p.shape[0])])
            p = p[order]
        s = self.renorm_parameterisation(p)
        
        p_interp = np.zeros(X.shape)
        s_interp = np.zeros(X.shape[0])
        d_sq_old = np.Inf
        
        for i in range(0, max_iter):
            s_interp, p_interp, d_sq = self.project(X, p, s)
            
            if np.abs(d_sq - d_sq_old) < tol:
                break
            d_sq_old = d_sq
            
            order = np.argsort(s_interp)
            # s_interp = s_interp[order]
            # X = X[order, :]

            s_in = len(s_interp)*self.s_factor

            spline = [UnivariateSpline(s_interp[order], X[order, j], s=s_in, k=self.k, w = w) for j in range(0, X.shape[1])] #Alter k, s

            p = np.zeros((len(s_interp), X.shape[1]))
            for j in range(0, X.shape[1]):
                p[:, j] = spline[j](s_interp[order])

            idx = [i for i in range(0, p.shape[0]-1) if (p[i] != p[i+1]).any()]
            p = p[idx, :]
            s = self.renorm_parameterisation(p)
            
        self.s = s
        self.p = p
        self.p_interp = p_interp
        self.s_interp = s_interp
        return
