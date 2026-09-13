"""
Visualization step methods for BaselineSolver.
Contains step_for_visualization and step_pure_profiled for real-time viewing and profiling.
"""

import time
import jax
import jax.numpy as jnp

# Import from local modules
from ..operators import (
    vorticity, vorticity_nonperiodic,
    divergence, divergence_nonperiodic,
    scalar_advection_diffusion_periodic, scalar_advection_diffusion_nonperiodic
)
from ..les_models import dynamic_smagorinsky, constant_smagorinsky
from ..boundary_conditions import apply_taylor_green_boundary_conditions
from ..pure_functions import apply_corner_smooth_pressure_gradient

# Import pressure solvers
try:
    from pressure_solvers.multigrid_solver_mac import poisson_multigrid_mac
    MAC_PRESSURE_AVAILABLE = True
except ImportError:
    MAC_PRESSURE_AVAILABLE = False

try:
    from pressure_solvers import poisson_multigrid, poisson_jacobi
except ImportError:
    poisson_multigrid = None
    poisson_jacobi = None

# Import MAC operators
try:
    from advection_schemes.rk3_mac import rk_step_unified_mac
    MAC_ADVECTION_AVAILABLE = True
except ImportError:
    MAC_ADVECTION_AVAILABLE = False

try:
    from solver.operators_mac import (
        divergence_staggered, divergence_nonperiodic_staggered,
        grad_x_staggered, grad_y_staggered,
        grad_x_nonperiodic_staggered, grad_y_nonperiodic_staggered,
        vorticity_nonperiodic_staggered, vorticity_staggered,
        scalar_advection_diffusion_periodic_staggered,
        scalar_advection_diffusion_nonperiodic_staggered
    )
    MAC_OPERATORS_AVAILABLE = True
except ImportError:
    MAC_OPERATORS_AVAILABLE = False

# Constants
rho = 1.0  # Fluid density


def step_pure_profiled(self, u, v, p, dt, iteration):
    """Non-JIT version of step_pure for detailed profiling of internal operations"""
    mask = self.mask
    dx, dy = self.grid.dx, self.grid.dy
    nu = self.flow.nu
    U_inf = self.flow.U_inf
    use_les = self.sim_params.use_les
    smagorinsky_constant = self.sim_params.smagorinsky_constant
    brinkman_eta = self.sim_params.brinkman_eta
    flow_type = self.sim_params.flow_type
    grid_type = self.sim_params.grid_type
    v_cycles = self.sim_params.multigrid_v_cycles
    fast_mode = getattr(self.sim_params, 'fast_mode', False)
    
    timing = {}
    
    # LES computation
    t0 = time.time()
    nu_total = nu
    nu_sgs_field = None
    if use_les:
        delta = (dx * dy) ** 0.5
        if grid_type == 'mac':
            from solver.operators_mac import interpolate_to_cell_center
            u_les, v_les = interpolate_to_cell_center(u, v)
        else:
            u_les, v_les = u, v
        if self.sim_params.les_model == 'dynamic_smagorinsky':
            nu_sgs_field, _ = dynamic_smagorinsky(u_les, v_les, dx, dy, delta)
        elif self.sim_params.les_model == 'smagorinsky':
            nu_sgs_field = constant_smagorinsky(u_les, v_les, dx, dy, delta, smagorinsky_constant)
            nu_total = nu  # Add SGS viscosity if needed
    timing['les_compute'] = time.time() - t0
    
    # Advection
    t0 = time.time()
    if grid_type == 'mac':
        from advection_schemes.rk3_mac import rk_step_unified_mac
        u_star, v_star = rk_step_unified_mac(u, v, dt, nu, dx, dy, mask, U_inf=U_inf,
                                             nu_sgs=nu_sgs_field, nu_hyper_ratio=0.0, slip_walls=True,
                                             fast_mode=fast_mode, brinkman_eta=brinkman_eta,
                                             flow_type=flow_type)
    else:
        from advection_schemes.rk3_simple_new import rk_step_unified
        u_star, v_star = rk_step_unified(u, v, dt, nu_total, dx, dy, mask, sdf=self.sdf, U_inf=U_inf,
                                         nu_sgs=None, nu_hyper_ratio=0.0, slip_walls=True,
                                         fast_mode=fast_mode, brinkman_eta=brinkman_eta,
                                         flow_type=flow_type)
    timing['advection'] = time.time() - t0
    
    # Divergence
    t0 = time.time()
    if grid_type == 'mac':
        from solver.operators_mac import divergence_staggered, divergence_nonperiodic_staggered
        if flow_type == 'von_karman' or flow_type == 'lid_driven_cavity':
            div_star = divergence_nonperiodic_staggered(u_star, v_star, dx, dy)
        else:
            div_star = divergence_staggered(u_star, v_star, dx, dy)
    else:
        if flow_type == 'von_karman' or flow_type == 'lid_driven_cavity':
            div_star = divergence_nonperiodic(u_star, v_star, dx, dy)
        else:
            div_star = divergence(u_star, v_star, dx, dy)
    rhs = div_star / dt
    timing['divergence'] = time.time() - t0
    
    # Pressure solve
    t0 = time.time()
    if grid_type == 'mac':
        from pressure_solvers.multigrid_solver_mac import poisson_multigrid_mac
        p = poisson_multigrid_mac(rhs, mask, dx, dy, v_cycles=v_cycles, flow_type=flow_type)
    else:
        p = poisson_multigrid(rhs, mask, dx, dy, v_cycles=v_cycles, flow_type=flow_type)
    timing['pressure_solve'] = time.time() - t0
    
    # Pressure gradient
    t0 = time.time()
    if grid_type == 'mac':
        from solver.operators_mac import grad_x_nonperiodic_staggered, grad_y_nonperiodic_staggered, grad_x_staggered, grad_y_staggered
        if flow_type == 'von_karman' or flow_type == 'lid_driven_cavity':
            dp_dx = grad_x_nonperiodic_staggered(p, dx)
            dp_dy = grad_y_nonperiodic_staggered(p, dy)
        else:
            dp_dx = grad_x_staggered(p, dx)
            dp_dy = grad_y_staggered(p, dy)
    else:
        if flow_type == 'von_karman':
            dp_dx = grad_x_nonperiodic(p, dx)
            dp_dy = grad_y_nonperiodic(p, dy)
        else:
            dp_dx = grad_x(p, dx)
            dp_dy = grad_y(p, dy)
    timing['pressure_gradient'] = time.time() - t0
    
    # Pressure correction
    t0 = time.time()
    u_new = u_star - dt * dp_dx
    v_new = v_star - dt * dp_dy
    timing['pressure_correction'] = time.time() - t0
    
    # Boundary conditions
    t0 = time.time()
    if flow_type == 'von_karman':
        # Apply inlet BC (excluding corners - wall BC takes precedence)
        u_new = u_new.at[0, 1:-1].set(U_inf)  # Inlet, excluding corners
        v_new = v_new.at[0, :].set(0.0)
        u_new = u_new.at[-1, :].set(u_new[-2, :])
        v_new = v_new.at[-1, :].set(v_new[-2, :])
        u_new = u_new.at[:, 0].set(u_new[:, 1])
        u_new = u_new.at[:, -1].set(u_new[:, -2])
        v_new = v_new.at[:, 0].set(0.0)
        v_new = v_new.at[:, -1].set(0.0)
    elif flow_type == 'lid_driven_cavity':
        cavity_height = dy * (u.shape[1] - 1)
        lid_velocity = 1.0
        u_new = u_new.at[:, -1].set(lid_velocity)
        u_new = u_new.at[:, 0].set(0.0)
        u_new = u_new.at[0, :].set(0.0)
        u_new = u_new.at[-1, :].set(0.0)
        v_new = v_new.at[:, 0].set(0.0)
        v_new = v_new.at[:, -1].set(0.0)
        v_new = v_new.at[0, :].set(0.0)
        v_new = v_new.at[-1, :].set(0.0)
    timing['boundary_conditions'] = time.time() - t0
    
    # Hard mask zeroing: force velocities to zero inside solid (mask < 0.5)
    # This is needed because mask has smooth sigmoid transition, so multiplication doesn't fully zero
    u_new = jnp.where(mask > 0.5, u_new, 0.0)
    v_new = jnp.where(mask > 0.5, v_new, 0.0)
    
    # Diagnostic: check velocity inside solid after mask application
    if self.iteration == 10:
        import numpy as np
        mask_np = np.array(mask)
        u_new_np = np.array(u_new)
        v_new_np = np.array(v_new)
        max_u_inside = np.max(np.where(mask_np < 0.5, np.abs(u_new_np), 0))
        max_v_inside = np.max(np.where(mask_np < 0.5, np.abs(v_new_np), 0))
        print(f"DEBUG after mask application: max |u| inside solid = {max_u_inside}, max |v| inside solid = {max_v_inside}")
    
    return u_new, v_new, p, timing


def step_for_visualization(self, compute_vorticity: bool = True, compute_divergence: bool = False, compute_drag_lift: bool = False, compute_diagnostics: bool = False):
    """
    Perform a single time step and return data for visualization.
    
    Args:
        compute_vorticity: Whether to compute vorticity field
        compute_divergence: Whether to compute divergence field
        compute_drag_lift: Whether to compute drag and lift (currently not implemented)
        compute_diagnostics: Whether to compute diagnostics (currently not implemented)
        
    Returns:
        u, v, vort, div (divergence may be None if not requested)
    """
    self.advance_steps(1)
    return self.visualization_fields(compute_vorticity, compute_divergence)


def advance_steps(self, count=1):
    """Advance on device; no field transfer or display work inside the loop.

    Adaptive mode applies conservative CFL/diffusion limits on every device step.
    """
    count = int(count)
    if count < 1:
        raise ValueError("count must be positive")
    if not 0 < float(self.dt) < float('inf'):
        raise ValueError("dt must be finite and positive")
    self._step_jit = self.get_step_jit()
    if getattr(self, '_batch_jit', None) is None:
        step = self._step_jit
        adaptive = self.sim_params.adaptive_dt
        h = min(self.grid.dx, self.grid.dy)
        speed = abs(float(self.flow.U_inf))
        velocity_cfl = .035 if speed > 8 else .05 if speed > 5 else .08 if speed > 3 else .12 if speed > 1.5 else .20
        from solver.params import get_re_parameters
        re_limits = get_re_parameters(self.flow.Re, self.grid.nx)
        cfl = min(float(self.sim_params.max_cfl), velocity_cfl, re_limits['cfl_target'])
        diffusion_dt = .20 * h * h / max(float(self.flow.nu), 1e-12)
        dt_max = float(self.sim_params.dt_max)
        @jax.jit
        def batch(u, v, p, mask, dt, iteration, n):
            def body(i, state):
                u, v, p, _, _, elapsed, used_dt = state
                if adaptive:
                    # Directional upper bound works for collocated and MAC shapes.
                    local_speed = jnp.maximum(jnp.max(jnp.abs(u)) + jnp.max(jnp.abs(v)), 1.5 * speed)
                    used_dt = jnp.minimum(cfl * h / jnp.maximum(local_speed, 1e-8),
                                          min(diffusion_dt, dt_max))
                un, vn, pn = step(u, v, mask, used_dt, iteration + i)
                return un, vn, pn, u, v, elapsed + used_dt, used_dt
            return jax.lax.fori_loop(0, n, body, (u, v, p, u, v, jnp.asarray(0., u.dtype), jnp.asarray(dt, u.dtype)))
        self._batch_jit = batch
    self.u, self.v, self.current_pressure, self.u_prev, self.v_prev, elapsed, last_dt = self._batch_jit(
        self.u, self.v, self.current_pressure, self.mask, self.dt, self.iteration, count)
    self.iteration += count
    if self.sim_params.adaptive_dt:
        elapsed, last_dt = jax.device_get((elapsed, last_dt))  # Two scalars, never full fields.
        self.dt = float(last_dt)
        advance = float(elapsed)
    else:
        advance = count * float(self.dt)
    self.simulated_time = getattr(self, 'simulated_time', 0.0) + advance
    if hasattr(self, 'state'):
        from dataclasses import replace
        self.state = replace(self.state, u=self.u, v=self.v, p=self.current_pressure,
                             u_prev=self.u_prev, v_prev=self.v_prev, c=self.c,
                             dt=self.dt, iteration=self.iteration)


def visualization_fields(self, compute_vorticity=True, compute_divergence=False):
    """Extract derived fields without advancing physical time."""
    vort = self._vorticity(self.u, self.v, self.grid.dx, self.grid.dy) if compute_vorticity else None
    div = self._divergence(self.u, self.v, self.grid.dx, self.grid.dy) if compute_divergence else None
    return self.u, self.v, vort, div
