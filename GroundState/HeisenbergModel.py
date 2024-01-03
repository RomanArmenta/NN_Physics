import numpy as np
import jax.numpy as jnp
import jax
import tools as t
import pickle

jax.config.update('jax_platform_name', 'cpu')

print("This script gets the ground-state energy of the one-dimensional Ising model with transversal field using Variational Monte Carlo and Restricted Boltzmann Machines.")
print("The Hamiltonian is: H = -g \sum(\sigmax_i) - J \sum(\sigmaz_i \sigmaz_i+1)")


n_sites = int(input("Write the number of spin sites: "))

J = float(input("Write J: "))
g = float(input("Write g: "))
n_iters = int(input("Write the number of Monte Carlo iterations: "))
n_samples = int(input("Write the number of samples to model the variational wavefunction: "))
n_ter = int(n_samples*float(input("Write the percentage of termalization of Markov chain: ")))
lr = float(input("Write the learning rate: "))
α_RBM = int(input("Write the RBMs alpha: "))
σ = float(input("Write the sampler's SD: "))

energies_sets = []
params_sets = []

prop = (J, g)
params_g = t.initializer_RBM(n_sites = n_sites, α = α_RBM, stdv = σ)
rng_key = jax.random.PRNGKey(np.random.choice(list(range(10_000))))

energies = []
for i in range(n_iters):
    rng_key, energy_step_g, samples_g, params_g = t.sgd_g(rng_key, params_g, n_samples, n_ter, lr, prop, n_sites)
    energies.append(energy_step_g)
    params_sets.append(params_g)
    print(i)
with open("./params_sets", 'wb') as file:
        pickle.dump(params_sets, file)

with open("./energies_sets", 'wb') as file:
        pickle.dump(energies_sets, file)