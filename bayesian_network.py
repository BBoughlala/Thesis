import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import TabularCPD


def generate_edges(n_var:int, p_edge:float) -> tuple:
    
    """
    Return a list of edges and nodes for a random Bayesian network.
        
        Parameters:
            n_var (int): Number of variables.
            p_edge (float): Probability of having an edge between two variables.
        
        Returns: 
            edges (list): List of tuples representing the edges between variables.
            nodes (list): List of nodes.
    """
    edges = []
    nodes = [v for v in range(n_var)]
    for i in range(n_var):
        for j in range(i+1, n_var):
            if np.random.rand() < p_edge:
                edges.append((i, j))
    return edges, nodes

def dependencies(var:int, edges:list) -> list:
    
    """
    Return a list of dependencies for a given variable.
        
        Parameters:
            var (int): Variable.
            edges (list): List of tuples representing the edges between variables.
        
        Returns: 
            deps (list): List of dependencies for the given variable.
    """
    deps = [v for v in range(var) if (v, var) in edges]
    return deps

def generate_cpd(nodes:list, edges:list) -> list:
        
        """
        Return a list of CPDs for a given network.

            Parameters:
                nodes (list): List of nodes.
                edges (list): List of tuples representing the edges between variables.
            
            Returns: 
                pds (list): List of CPDs for the given network.
        """
        cpds = []
        for i in nodes:
            deps = dependencies(i, edges)
            if deps:
                n_rows = 2
                n_cols = 2**len(deps)
                cpd = np.random.rand(n_rows, n_cols)
                cpd = cpd / cpd.sum(axis=0)
                cpds.append(TabularCPD(variable=i, variable_card=2, values=cpd, evidence=deps, evidence_card=[2]*len(deps)))
            else:
                cpd = np.random.rand(2)
                cpd = cpd / cpd.sum()
                cpd = cpd.reshape(2, 1)
                cpds.append(TabularCPD(variable=i, variable_card=2, values=cpd))
        return cpds

def generate_network(nodes:list, edges:list, cpds:list) -> BayesianNetwork:
    
    """
    Return a Bayesian network.

        Parameters:
            nodes (list): List of nodes.
            edges (list): List of tuples representing the edges between variables.
            cpds (list): List of CPDs for the given network.

        Returns: 
            model (BayesianNetwork): The Bayesian network.
    """
    model = BayesianNetwork()
    model.add_nodes_from(nodes)
    model.add_edges_from(edges)
    for cpd in cpds:
        model.add_cpds(cpd)
    return model

def generate_samples(network:BayesianNetwork, n_samples:int) -> pd.DataFrame:
    
    """
    Return a DataFrame of samples from a given Bayesian network.
        
        Parameters:
            network (BayesianNetwork): The Bayesian network.
            n_samples (int): Number of samples to generate.
        
        Returns: 
            samples (pd.DataFrame): The generated samples.
    """
    inference = BayesianModelSampling(network)
    samples = inference.forward_sample(size=n_samples)
    return samples


class BayesianNetworkSampler():
    """
    A class to generate samples from a Bayesian network.

    ...

    Attributes
    ----------
    n_var : int
        Number of variables.
    p_edge : float
        Probability of having an edge between two variables.
    edges : list
        List of tuples representing the edges between variables.
    nodes : list
        List of nodes.
    cpds : list
        List of CPDs for the given network.
    network : BayesianNetwork
        The Bayesian network.
    
    Methods
    -------
    sample(n_samples:int) -> pd.DataFrame:
        Generate samples from the Bayesian network.
    """
    def __init__(self, n_var:int, p_edge) -> None:
        self.n_var = n_var
        self.p_edge = p_edge
        self.edges, self.nodes = generate_edges(self.n_var, self.p_edge)
        self.cpds = generate_cpd(self.nodes, self.edges)
        self.network = generate_network(self.nodes, self.edges, self.cpds)

    def sample(self, n_samples:int) -> pd.DataFrame:
        """
        Generate samples from the Bayesian network.

        Parameters:
            n_samples (int): Number of samples to generate.
        
        Returns:
            samples (pd.DataFrame): The generated samples.
        """
        return generate_samples(self.network, n_samples)