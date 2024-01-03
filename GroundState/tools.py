import numpy as np
import jax  
import jax.numpy as jnp
from functools import partial 

def initializer_RBM(n_sites, α = 1, stdv = 0.01, complex_dtype = False):
    if complex_dtype:
        a_ = jnp.asarray(np.random.normal(size = n_sites, scale = stdv) + 1j*np.random.normal(size = n_sites, scale = stdv))
        b_ = jnp.asarray(np.random.normal(size = int(α*n_sites), scale = stdv) + 1j*np.random.normal(size = int(α*n_sites), scale = stdv))
        W_ = jnp.asarray(np.random.normal(size = (int(α*n_sites), n_sites), scale = stdv) + 1j*np.random.normal(size = (int(α*n_sites), n_sites), scale = stdv))
    else:
        a_ = jnp.asarray(np.random.normal(size = n_sites, scale = stdv))
        b_ = jnp.asarray(np.random.normal(size = int(α*n_sites), scale = stdv))
        W_ = jnp.asarray(np.random.normal(size = (int(α*n_sites), n_sites), scale = stdv))
    return [a_, b_, W_]

@jax.jit
def probRBM(params_, v_): 
    a_, b_, W_= params_
    exponent_ = jnp.dot(a_, v_) 
    c_ = b_ + jnp.dot(W_, v_)
    prod_ = jnp.prod(2*jnp.cosh(c_))
    #prod_ = jnp.prod(c_)
    return jnp.exp(exponent_)*prod_

@jax.jit
def wf_g(params_g, v_):
    return probRBM(params_g, v_)

@partial(jax.vmap, in_axes = (None, 0), out_axes = 0)
@jax.jit
def wf_squared_g(params_g, v_):
    return wf_g(params_g, v_)**2

@jax.jit
def log_wf_g(params_g, v_):
    return jnp.log(wf_g(params_g, v_))

@jax.jit
def log_pdf_g(params, v_):
    return jnp.log((wf_g(params, v_))**2)

log_grads_g = jax.grad(log_wf_g)

@partial(jax.jit, static_argnums=(1,2))
def rw_metropolis_kernel(rng_key, log_pdf, n_sites, params, position):
    logpdf = partial(log_pdf, params)
    rng_key, key1, key2 = jax.random.split(rng_key, 3)
    site_proposed = jax.random.choice(key1, jnp.array(list(range(n_sites))))
    proposal = position.at[site_proposed].multiply(-1)
    log_wf = logpdf(position)
    proposal_log_wf = logpdf(proposal)

    log_uniform = jnp.log(jax.random.uniform(key2))
    do_accept = log_uniform < proposal_log_wf - log_wf

    position = jnp.where(do_accept, proposal, position)
    return rng_key, position

@partial(jax.jit, static_argnums=(1,2,3))
def mh_sampler_jax(rng_key, n_samples, log_pdf, n_sites, params):
    
    def mh_step(carry, x):
        rng_key, position = carry
        rng_key, position = rw_metropolis_kernel(rng_key, log_pdf, n_sites, params, position)
        return (rng_key, position), position
    
    rng_key, rng_key_2, rng_key_3 = jax.random.split(rng_key, 3)
    carry = (rng_key_3, jax.random.choice(rng_key_2, jnp.array([-1, 1]), shape = (n_sites,)))
    _, samples = jax.lax.scan(mh_step, carry, None, n_samples)
    return rng_key, samples

@partial(jax.vmap, in_axes = (None, None, 0, None), out_axes = 0)
def en_loc_g(params, prop, pos, n_sites):
    J, g = prop
    func = partial(wf_g, params)
    local = jnp.sum(pos*jnp.roll(pos, 1))*(-J) + \
                jnp.sum(jnp.array([func(pos.at[i].multiply(-1)) for i in range(n_sites)]))*(-g)/func(pos)
    return local

@partial(jax.vmap, in_axes = (None, 0, 0, None), out_axes = 0)
def grad_en_g(params_g, sample, loc_en, sv_en):
    grads = jax.grad(log_wf_g)(params_g, sample)
    return [2*jnp.real(((loc_en - sv_en))*grad) for grad in grads]

def sgd_g(rng_key, params, n_samples, n_ter, lr, prop, n_sites):
    rng_key, samples_ = mh_sampler_jax(rng_key, n_samples + n_ter, log_pdf_g, n_sites, params)
    samples = samples_[n_ter:]
    samples_array = jnp.array(samples)
    loc_en = en_loc_g(params, prop, samples_array, n_sites)
    sv_en = jnp.mean(loc_en)
    grads = grad_en_g(params, samples, loc_en, sv_en)
    return rng_key, sv_en, samples_array, [p - lr*jnp.mean(grad, axis = 0) for p, grad in zip(params, grads)]

