from brainlit.algorithms.trace_analysis.fit_spline import GeometricGraph
import typing
import numpy as np

class Diffeomorphism():
    def __init__(self):
        pass 

    def evaluate(position):
        return position
    
    def D(position, order: int = 1):
        return np.eye(3)

def transform_GeometricGraph(G: GeometricGraph, Phi: Diffeomorphism):
