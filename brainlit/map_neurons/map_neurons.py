from brainlit.algorithms.trace_analysis.fit_spline import GeometricGraph
import typing
import numpy as np
import copy
import networkx as nx
from scipy.interpolate import BSpline, splev, CubicHermiteSpline, RegularGridInterpolator
import h5py
from tqdm import tqdm


class DiffeomorphismAction:
    def evaluate(self, position: np.array) -> np.array:
        pass

    def D(self, position: np.array, deriv: np.array, order: int = 1) -> np.array:
        pass


class CloudReg_Transform(DiffeomorphismAction):
    def __init__(self, vpath):
        f = h5py.File(vpath,'r')
        self.vtx = np.array(f.get('vtx'))
        self.vty = np.array(f.get('vty'))
        self.vtz = np.array(f.get('vtz'))

        A = np.zeros((3, 4))
        A[:3,:3] = np.eye(3)
        self.A = A

        self._integrate()

    def _integrate(self, velocity_voxel_size = [100., 100., 100.]):
        # translated from https://github.com/neurodata/CloudReg/blob/master/cloudreg/registration/transform_points.m
        vtx = self.vtx
        vty = self.vty
        vtz = self.vtz
        A = self.A

        nT = vtx.shape[0]
        dt = 1/nT 

        nxV = vtx.shape[1:]
        dxV = velocity_voxel_size
        xV = np.arange(0,nxV[0])*dxV[0]
        yV = np.arange(0,nxV[1])*dxV[1]
        zV = np.arange(0,nxV[2])*dxV[2]
        xV = xV - np.mean(xV)
        yV = yV - np.mean(yV)
        zV = zV - np.mean(zV)

        [XV,YV,ZV] = np.meshgrid(xV,yV,zV, indexing='ij')
        timesteps = np.arange(nT-1,-1, -1)
        indicator = 1

        transx = XV
        transy = YV
        transz = ZV

        for t in tqdm(timesteps, desc='integrating velocity field'):
            Xs = XV + indicator*vtx[t,:,:,:]*dt
            Ys = YV + indicator*vty[t,:,:,:]*dt
            Zs = ZV + indicator*vtz[t,:,:,:]*dt
            F = RegularGridInterpolator((xV,yV,zV),transx-XV,method='linear', bounds_error=False)
            XYZs = np.reshape(np.stack((Xs,Ys,Zs), axis=-1), newshape=(-1,3))
            transx = np.reshape(F(XYZs), Xs.shape) + Xs
            F = RegularGridInterpolator((xV,yV,zV),transy-YV,method='linear', bounds_error=False)
            transy = np.reshape(F(XYZs), Ys.shape) + Ys
            F = RegularGridInterpolator((xV,yV,zV),transz-ZV,method='linear', bounds_error=False)
            transz = np.reshape(F(XYZs), Zs.shape) + Zs

        Atransx = A[0,0]*transx + A[0,1]*transy + A[0,2]*transz + A[0,3]
        Atransy = A[1,0]*transx + A[1,1]*transy + A[1,2]*transz + A[1,3]
        Atransz = A[2,0]*transx + A[2,1]*transy + A[2,2]*transz + A[2,3]


        Fx = RegularGridInterpolator((xV,yV,zV),Atransx,method='linear', bounds_error=False)
        Fy = RegularGridInterpolator((xV,yV,zV),Atransy,method='linear', bounds_error=False)
        Fz = RegularGridInterpolator((xV,yV,zV),Atransz,method='linear', bounds_error=False)

        self.diffeomorphism = (Fx, Fy, Fz)
        
    def evaluate(self, position: np.array) -> np.array:
        transformed_positionx = self.diffeomorphism[0](position[:,0],position[:,1],position[:,2])
        transformed_positiony = self.diffeomorphism[1](position[:,0],position[:,1],position[:,2])
        transformed_positionz = self.diffeomorphism[2](position[:,0],position[:,1],position[:,2])

        transformed_position = np.stack((transformed_positionx,transformed_positiony,transformed_positionz), axis=0)

        return transformed_position


    def D(self, position: np.array, deriv: np.array, order: int = 1) -> np.array:
        diffx = self.diffeomorphism[0]
        diffy = self.diffeomorphism[0]
        diffz = self.diffeomorphism[0]

        transformed_deriv = deriv.copy()
        step = 1
        for i, (pos, d) in enumerate(zip(position, deriv)):
            J = np.zeros((3,3))
            J[0,0] = (diffx(pos[0]+step,pos[1],position[2]) - diffx(pos[:,0],pos[:,1],pos[:,2]))/step
            J[0,1] = (diffx(pos[0],pos[1]+step,position[2]) - diffx(pos[:,0],pos[:,1],pos[:,2]))/step
            J[0,2] = (diffx(pos[0],pos[1],position[2]+step) - diffx(pos[:,0],pos[:,1],pos[:,2]))/step

            J[1,0] = (diffy(pos[0]+step,pos[1],position[2]) - diffy(pos[:,0],pos[:,1],pos[:,2]))/step
            J[1,1] = (diffy(pos[0],pos[1]+step,position[2]) - diffy(pos[:,0],pos[:,1],pos[:,2]))/step
            J[1,2] = (diffy(pos[0],pos[1],position[2]+step) - diffy(pos[:,0],pos[:,1],pos[:,2]))/step

            J[2,0] = (diffx(pos[0]+step,pos[1],position[2]) - diffx(pos[:,0],pos[:,1],pos[:,2]))/step
            J[2,1] = (diffx(pos[0],pos[1]+step,position[2]) - diffx(pos[:,0],pos[:,1],pos[:,2]))/step
            J[2,2] = (diffx(pos[0],pos[1],position[2]+step) - diffx(pos[:,0],pos[:,1],pos[:,2]))/step
            transformed_deriv[i,:] = np.matmul(J, d).T









def transform_GeometricGraph(G: GeometricGraph, Phi: DiffeomorphismAction):
    if G.spline_type is not BSpline:
        raise NotImplementedError("Can only transform bsplines")

    if not G.spline_tree:
        G.fit_spline_tree_invariant()
    G_tranformed = copy.deepcopy(G)

    spline_tree = G_tranformed.spline_tree

    # process in reverse dfs order to ensure parents are processed after
    reverse_dfs = list(reversed(list(nx.topological_sort(spline_tree))))

    for node in reverse_dfs:
        path = spline_tree.nodes[node]["path"]
        tck, us = spline_tree.nodes[node]["spline"]
        positions = np.array(splev(us, tck, der=0)).T
        transformed_positions = Phi.evaluate(positions)
        derivs = np.array(splev(us, tck, der=1)).T
        transformed_derivs = Phi.D(positions, derivs, order=1)

        chspline = CubicHermiteSpline(us, transformed_positions, transformed_derivs)

        spline_tree.nodes[node]["spline"] = chspline, us

        for trace_node, position, deriv in zip(
            path, transformed_positions, transformed_derivs
        ):
            G_tranformed.nodes[trace_node]["loc"] = position
            G_tranformed.nodes[trace_node]["deriv"] = deriv

    return G_tranformed
