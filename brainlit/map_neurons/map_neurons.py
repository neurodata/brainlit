from brainlit.algorithms.trace_analysis.fit_spline import GeometricGraph, compute_parameterization
import typing
import numpy as np
import copy
import networkx as nx
from scipy.interpolate import BSpline, splev, CubicHermiteSpline, RegularGridInterpolator
import h5py
from tqdm import tqdm
import scipy.io as io


class DiffeomorphismAction:
    def evaluate(self, position: np.array) -> np.array:
        pass

    def D(self, position: np.array, deriv: np.array, order: int = 1) -> np.array:
        pass


class CloudReg_Transform(DiffeomorphismAction):
    """Object that can read mat files from CloudReg, and compute transformations on points and Jacobians.
    Implements DiffeomorphismAction which is an interface to transform points and tangent vectors.
    """
    def __init__(self, vpath, Apath):
        """Compute transformation from CloudReg mat files

        Args:
            vpath (str): path to mat file
        """
        # not: transformation files go from template space to target space
        f = h5py.File(vpath,'r')
        self.vtx = np.array(f.get('vtx'))
        self.vty = np.array(f.get('vty'))
        self.vtz = np.array(f.get('vtz'))

        f = io.loadmat(Apath)
        A = f['A']
        self.A = A
        self.B = np.linalg.inv(A)

        self._integrate()

    def apply_affine(self, position: np.array) -> np.array:
        # transformation direction: atlas
        A = self.B
        transformed_position = A[:3,:3] @ position.T + np.expand_dims(A[:3,3], axis=0).T
        transformed_position = transformed_position.T

        return transformed_position


    def _integrate(self, velocity_voxel_size = [100., 100., 100.]):
        # translated from https://github.com/neurodata/CloudReg/blob/master/cloudreg/registration/transform_points.m
        # transformation direction: atlas

        vtx = self.vtx
        vty = self.vty
        vtz = self.vtz

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
        timesteps = np.arange(0,nT, 1)
        indicator = - 1

        transx = XV
        transy = YV
        transz = ZV

        for t in tqdm(timesteps, desc='integrating velocity field'):
            Xs = XV + indicator*vtx[t,:,:,:]*dt
            Ys = YV + indicator*vty[t,:,:,:]*dt
            Zs = ZV + indicator*vtz[t,:,:,:]*dt
            F = RegularGridInterpolator((xV,yV,zV),transx-XV,method='linear', bounds_error=False, fill_value=None)
            XYZs = np.reshape(np.stack((Xs,Ys,Zs), axis=-1), newshape=(-1,3))
            transx = np.reshape(F(XYZs), Xs.shape) + Xs
            F = RegularGridInterpolator((xV,yV,zV),transy-YV,method='linear', bounds_error=False, fill_value=None)
            transy = np.reshape(F(XYZs), Ys.shape) + Ys
            F = RegularGridInterpolator((xV,yV,zV),transz-ZV,method='linear', bounds_error=False, fill_value=None)
            transz = np.reshape(F(XYZs), Zs.shape) + Zs

        Fx = RegularGridInterpolator((xV,yV,zV),transx-XV,method='linear', bounds_error=False, fill_value=None)
        Fy = RegularGridInterpolator((xV,yV,zV),transy-YV,method='linear', bounds_error=False, fill_value=None)
        Fz = RegularGridInterpolator((xV,yV,zV),transz-ZV,method='linear', bounds_error=False, fill_value=None)

        self.og_coords = [XV,YV,ZV]
        self.diffeomorphism = (Fx, Fy, Fz)
        
    def evaluate(self, position: np.array) -> np.array:
        Fx = self.diffeomorphism[0]
        Fy = self.diffeomorphism[1]
        Fz = self.diffeomorphism[2]

        transformed_positionx = Fx(position) + position[:,0]
        transformed_positiony = Fy(position) + position[:,1]
        transformed_positionz = Fz(position) + position[:,2]

        transformed_position = np.stack((transformed_positionx,transformed_positiony,transformed_positionz), axis=1)

        return transformed_position


    def D(self, position: np.array, deriv: np.array, order: int = 1) -> np.array:
        Fx = self.diffeomorphism[0]
        Fy = self.diffeomorphism[1]
        Fz = self.diffeomorphism[2]

        transformed_deriv = deriv.copy()
        step = 1
        for i, (pos, d) in enumerate(zip(position, deriv)):
            J = np.zeros((3,3))
            J[0,0] = (Fx([pos[0]+step,pos[1],pos[2]]) + step - Fx(pos))/step
            J[0,1] = (Fx([pos[0],pos[1]+step,pos[2]]) - Fx(pos))/step
            J[0,2] = (Fx([pos[0],pos[1],pos[2]+step]) - Fx(pos))/step

            J[1,0] = (Fy([pos[0]+step,pos[1],pos[2]]) - Fy(pos))/step
            J[1,1] = (Fy([pos[0],pos[1]+step,pos[2]]) + step - Fy(pos))/step
            J[1,2] = (Fy([pos[0],pos[1],pos[2]+step]) - Fy(pos))/step

            J[2,0] = (Fz([pos[0]+step,pos[1],pos[2]]) - Fz(pos))/step
            J[2,1] = (Fz([pos[0],pos[1]+step,pos[2]]) - Fz(pos))/step
            J[2,2] = (Fz([pos[0],pos[1],pos[2]+step]) + step - Fz(pos))/step
            transformed_deriv[i,:] = np.matmul(J, d).T
        
        return transformed_deriv


def transform_GeometricGraph(G: GeometricGraph, Phi: DiffeomorphismAction, deriv_method="spline"):
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
        transformed_us = compute_parameterization(transformed_positions)
        if deriv_method == "spline":
            derivs = np.array(splev(us, tck, der=1)).T
        elif deriv_method == "difference":
            # Sundqvist & Veronis 1970
            f_im1 = transformed_positions[:-2,:]
            f_i =  transformed_positions[1:-1,:]
            f_ip1 = transformed_positions[2:,:]
            hs = np.diff(transformed_us)
            h_im1 = np.expand_dims(hs[:-1], axis=1)
            h_i =  np.expand_dims(hs[1:], axis=1)

            if len(transformed_us) >= 3:
                diffs = f_ip1 - np.multiply((1 - np.divide(h_i, h_im1) ** 2), f_i) - np.multiply(np.divide(h_i, h_im1)**2, f_im1)
                diffs = np.concatenate(([transformed_positions[1,:]-transformed_positions[0,:]] , diffs, [transformed_positions[-1,:]-transformed_positions[-2,:]]), axis=0)
            elif len(transformed_us) == 2:
                diffs = np.array([transformed_positions[1,:]-transformed_positions[0,:], transformed_positions[-1,:]-transformed_positions[-2,:]])
            norms = np.linalg.norm(diffs, axis=1)
            derivs = np.divide(diffs, np.array([norms]).T)


        else:
            raise ValueError(f"Invalid deriv_method argument: {deriv_method}")

        transformed_derivs = Phi.D(positions, derivs, order=1)

        chspline = CubicHermiteSpline(transformed_us, transformed_positions, transformed_derivs)

        spline_tree.nodes[node]["spline"] = chspline, transformed_us

        for trace_node, position, deriv in zip(
            path, transformed_positions, transformed_derivs
        ):
            G_tranformed.nodes[trace_node]["loc"] = position
            G_tranformed.nodes[trace_node]["deriv"] = deriv

    return G_tranformed
