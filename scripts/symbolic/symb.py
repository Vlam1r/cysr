import os
from functools import partial

import jax
import jax.numpy as jnp
from jax_autovmap import auto_vmap

import cyjax

os.environ["PATH"] += ":/home/vlamir/julia-1.9.0-beta4/bin"

import numpy as np
import sympy
from pysr import PySRRegressor


import pandas
import pickle
import scipy


@auto_vmap(params=1, h=2)
@partial(jax.jit, backend='cpu', static_argnames=['count', 'metric'])
def create_samples(metric, key, params, h, count):
    zs, patch = metric.variety.sample_intersect(key, params, count, affine=True)
    g = metric.metric(h, zs, params, patch)
    ricci = metric.ricci_scalar(h, zs, params, patch)
    kahler = metric.kahler_potential(h, zs, params, patch)
    return zs, patch, g, ricci, kahler


@auto_vmap(params=1, h=2)
@partial(jax.jit, backend='cpu', static_argnames=['count', 'metric'])
def create_samples_funni(metric, key, params, h, count):
    # zs, patch = metric.variety.sample_intersect(key, params, count, affine=True)
    c = (1+1j)/jnp.sqrt(2)
    z1 = jax.random.ball(key, d=2, shape=(count, ))
    z1 = z1[:,0] + 1j * z1[:,1]

    z2 = jnp.concatenate([c*z1, c**3 * z1, c**5 * z1, c**7 * z1])
    z1 = jnp.concatenate([z1, z1, z1, z1])
    z0 = c*jnp.ones_like(z1, dtype=jnp.complex64)

    zs = jnp.stack([z0, z1, z2], axis=1)
    patch = jnp.zeros_like(z1, dtype=jnp.int32)

    g = metric.metric(h, zs, params, patch)
    ricci = metric.ricci_scalar(h, zs, params, patch)
    kahler = metric.kahler_potential(h, zs, params, patch)
    return zs, patch, g, ricci, kahler


def sample_points(model, metric, params, dim,
                  threshold=1e-5, count=10_000, rng=jax.random.PRNGKey(42), batch=1000000):
    zs_out = jnp.zeros(shape=(0, dim))
    g_out = jnp.zeros(shape=(0, dim-1, dim-1))
    ricci_out = jnp.zeros(shape=(0,))
    while zs_out.shape[0] < count:
        rng, new_rng = jax.random.split(rng)
        h = model.apply(params, 0 + 0j, deterministic=True)
        h = cyjax.ml.cholesky_from_param(h)
        zs, patch, g, ricci, _ = create_samples_funni(metric, new_rng, [0 + 0j], h, batch)
        # Same patch and dep for consistency
        # Small values
        g_dep = g[2]
        filter = (patch == 0) & (g_dep == 0) & (jnp.abs(ricci) < threshold)
        zs = zs[filter]
        g = g[0][filter]
        ricci = ricci[filter].real
        # Add to total
        zs_out = jnp.concatenate([zs_out, zs], axis=0)
        g_out = jnp.concatenate([g_out, g], axis=0)
        ricci_out = jnp.concatenate([ricci_out, ricci], axis=0)
        print(f"Found {zs_out.shape[0]} of {count}")
    return zs_out, g_out, ricci_out


def main():
    saved = np.load('../k3_samples.npz')

    zs = saved['zs']
    g = saved['g']
    # ricci = saved['ricci']

    model = PySRRegressor(
        procs=16,
        populations=40,
        niterations=1000000,  # Infty
        timeout_in_seconds=60 * 60 * 3,
        # batching=True,
        # batch_size=100,
        maxsize=100,
        maxdepth=50,
        ncyclesperiteration=2500,  # default 550
        binary_operators=["+", '-', "*"],
        unary_operators=['inv(x)=1/x'],
        # complexity_of_variables=1.01,
        complexity_of_constants=15,
        complexity_of_operators={'inv':3},
        # constraints={'^':(-1,1)},
        # select_k_features=12,
        progress=True,
        warm_start=True,
        turbo=True,  # Faster evaluation (experimental)
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        )

    # Feature generation
    X = np.concatenate([zs.real[:,:],  # x0 - x2
                        zs.imag[:,:],  # x3 - x5
                        jnp.abs(zs)[:,:],  # x6 - x8
                        (zs[:,1] * zs[:,2]).real.reshape(-1,1),
                        (zs[:,1] * zs[:,2]).imag.reshape(-1,1),
                        (zs[:, 1] * jnp.conj(zs[:, 2])).real.reshape(-1, 1),
                        (zs[:, 1] * jnp.conj(zs[:, 2])).imag.reshape(-1, 1),
                        (zs[:, 1] ** 2 / zs[:, 0] ** 2).real.reshape(-1,1) ** 2,
                        ], axis=1)

    # X = np.stack([zs[:,1].real,
    #             zs[:,1].imag,
    #             jnp.abs(zs)[:,1],
    #             (zs[:, 1]**2).real,
    #             (zs[:, 1]**2).imag,
    #             ], axis=1)

    # y = np.array([g[:,0,0].real, g[:,0,1].real, g[:,0,1].imag]).transpose()

    y = g[:,0,0].real

    # filter =  jnp.abs(zs)[:,0] > 0.99
    # X = X[filter, ...]
    # y = y[filter, ...]

    # Size limit
    X = X[:2000, ...]
    y = y[:2000, ...]

    print(f"Running with {X.shape[0]} points...")
    model.fit(X, y)

    pandas.set_option('display.max_columns', None)
    print(model)
    with open('eqs.pkl', 'wb') as file:
      pickle.dump(model.equations_, file)

if __name__ == '__main__':
    main()