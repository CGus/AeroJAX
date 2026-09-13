"""
JAX Configuration

Configure JAX settings before any JAX imports.
"""

import jax


def configure_jax(platform='cpu', enable_x64=False, debug_nans=False):
    """
    Configure JAX global settings.
    
    Args:
        platform: 'cpu', 'gpu', or 'tpu'
        enable_x64: Enable 64-bit floats
        debug_nans: Enable NaN debugging
    """
    jax.config.update('jax_platform_name', platform)
    jax.config.update('jax_enable_x64', enable_x64)
    jax.config.update('jax_debug_nans', debug_nans)


# Backend selection belongs to the entry point/JAX environment, not an import.
# JAX selects an available accelerator unless the caller explicitly selects CPU.


def invalidate_solver_cache(solver):
    """Keep global cache eviction only for the unaudited legacy LBM backend."""
    if getattr(solver.sim_params, 'solver_type', 'navier_stokes') == 'lattice_boltzmann':
        jax.clear_caches()
    if hasattr(solver, '_jit_cache'):
        solver._jit_cache.clear()
    if hasattr(solver, '_batch_jit'):
        solver._batch_jit = None
