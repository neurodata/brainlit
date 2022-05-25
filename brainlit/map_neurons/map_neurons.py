from brainlit.algorithms.trace_analysis.fit_spline import GeometricGraph
import typing
import numpy as np
import copy
import networkx as nx
from scipy.interpolate import BSpline, splev, CubicHermiteSpline

class DiffeomorphismAction():

    def evaluate(self, position: np.array) -> np.array:
        pass
    
    def D(self, position: np.array, deriv: np.array, order: int = 1) -> np.array:
        pass

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
        positions = np.array(splev(us, tck, der = 0)).T
        transformed_positions = Phi.evaluate(positions)
        derivs = np.array(splev(us, tck, der = 1)).T
        transformed_derivs = Phi.D(positions, derivs, order=1)

        chspline = CubicHermiteSpline(us, transformed_positions, transformed_derivs)

        spline_tree.nodes[node]["spline"] = chspline, us

        for trace_node, position, deriv in zip(path, transformed_positions, transformed_derivs):
            G_tranformed.nodes[trace_node]["loc"] = position
            G_tranformed.nodes[trace_node]["deriv"] = deriv

    return G_tranformed

