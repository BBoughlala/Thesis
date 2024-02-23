from bayesian_network import BayesianNetworkSampler
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Generate samples from a Bayesian network.')

parser.add_argument('--min_n_var', type=int, help='Number of variables.', required=True)
parser.add_argument('--max_n_var', type=int, help='Number of variables.', required=True)

parser.add_argument('--min_p_edge', type=float, help='Probability of having an edge between two variables.', required=True)
parser.add_argument('--max_p_edge', type=float, help='Probability of having an edge between two variables.', required=True)

parser.add_argument('--n_samples', type=int, help='Number of samples to generate.', required=True)
parser.add_argument('--output_loc', type=str, help='Output file.', required=True)

args = parser.parse_args()

for n_var in range(args.min_n_var, args.max_n_var + 1):
    for p_edge in np.arange(args.min_p_edge, args.max_p_edge + 0.1, 0.1):
        sampler = BayesianNetworkSampler(n_var=n_var, p_edge=p_edge)
        samples = sampler.sample(n_samples=args.n_samples)
        samples.to_csv(f'{args.output_loc}\V{n_var}P{p_edge:.1f}.csv', index=False)