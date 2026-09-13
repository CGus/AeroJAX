"""
Simulation controller for Baseline Navier-Stokes Viewer
Handles simulation threading, data management, and worker communication
"""

import copy
import sys
import time
import threading
import queue
import logging
import numpy as np
import jax.numpy as jnp
import jax
import multiprocessing.shared_memory as shm
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QTimer, QCoreApplication, Qt

logger = logging.getLogger(__name__)


class SharedData:
    """Shared memory buffer for zero-copy data transfer"""
    def __init__(self, shape, dtype=np.float32):
        size = int(np.prod(shape) * np.dtype(dtype).itemsize)  # Convert to Python int
        self.shm = shm.SharedMemory(create=True, size=size)
        self.array = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)
    
    def cleanup(self):
        """Idempotently close and unlink shared memory."""

        if getattr(self, "_cleaned", False):
            return

        self._cleaned = True

        shm_obj = getattr(self, "shm", None)

        if shm_obj is None:
            return

        try:
            shm_obj.close()
        except (FileNotFoundError, BufferError):
            pass
        except Exception as exc:
            logger.debug(
                f"Shared memory close ignored: {exc}"
            )

        try:
            shm_obj.unlink()
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.debug(
                f"Shared memory unlink ignored: {exc}"
            )

class MetricsWorker(QObject):
    """Separate thread for metrics computation to avoid blocking simulation"""
    metrics_ready = pyqtSignal(dict)  # Signal when new metrics are ready

    def __init__(self, solver):
        super().__init__()
        self.solver = solver
        self.running = False
        self.paused = False
        self.data_queue = queue.Queue(maxsize=2)  # Buffer of 2 frames
        self.thread = None
        self.frame_count = 0

    def start(self):
        """Start the metrics worker exactly once."""

        if (
            self.thread is not None
            and self.thread.is_alive()
        ):
            logger.debug(
                "Metrics worker start ignored: already running"
            )
            return

        self.running = True
        self.paused = False

        self.thread = threading.Thread(
            target=self.run_metrics,
            daemon=True,
            name="AeroJAX-MetricsWorker"
        )

        self.thread.start()

        logger.info("Metrics worker started")

    def stop(self):
        """Stop the metrics worker safely."""

        thread = self.thread

        if (
            not self.running
            and (
                thread is None
                or not thread.is_alive()
            )
        ):
            return

        self.running = False
        self.paused = False

        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=1.0)

        if thread is not None and thread.is_alive():
            raise RuntimeError("Metrics worker is still stopping; state was retained")
        self.thread = None
        with self.data_queue.mutex:
            self.data_queue.queue.clear()
        logger.info("Metrics worker stopped")

    def pause(self):
        """Pause the metrics computation"""
        self.paused = True

    def resume(self):
        """Resume the metrics computation"""
        self.paused = False

    def enqueue_data(self, u, v, pressure, mask, iteration):
        """Enqueue simulation data for metrics computation"""
        try:
            if self.data_queue.full():
                # Drop oldest frame if queue is full
                try:
                    self.data_queue.get_nowait()
                except queue.Empty:
                    pass
            snapshot = copy.copy(self.solver)
            snapshot.flow = copy.copy(self.solver.flow)
            snapshot.sim_params = copy.copy(self.solver.sim_params)
            self.data_queue.put((u, v, pressure, mask, iteration, snapshot), block=False)
        except queue.Full:
            pass  # Drop frame if queue is full

    def run_metrics(self):
        """Main loop for metrics computation"""
        import numpy as np
        from solver.metrics import find_stagnation_point, find_separation_point, compute_forces_ibm, get_airfoil_surface_mask
        from solver.operators import laplacian_nonperiodic_x, divergence_nonperiodic

        while self.running:
            if self.paused:
                time.sleep(0.01)
                continue

            try:
                # Get data from queue with timeout
                u, v, pressure, mask, iteration, solver = self.data_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                # Convert JAX arrays to numpy for computation
                u_np = np.array(u)
                v_np = np.array(v)
                pressure_np = np.array(pressure)
                mask_np = np.array(mask)
                
                dx, dy = solver.grid.dx, solver.grid.dy
                # For MAC grid, interpolate velocities to cell centers before computing metrics
                grid_type = getattr(solver.sim_params, 'grid_type', 'collocated')
                # Save original staggered arrays for divergence computation (convert to numpy first)
                u_np_staggered = np.array(u_np) if grid_type == 'mac' else None
                v_np_staggered = np.array(v_np) if grid_type == 'mac' else None
                if grid_type == 'mac':
                    # u is staggered in x (nx+1, ny), interpolate in x: 0.5 * (u[:-1, :] + u[1:, :])
                    u_np = 0.5 * (u_np[:-1, :] + u_np[1:, :])
                    # v is staggered in y (nx, ny+1), interpolate in y: 0.5 * (v[:, :-1] + v[:, 1:])
                    v_np = 0.5 * (v_np[:, :-1] + v_np[:, 1:])

                # Check if we should compute metrics based on frame skip (match solver behavior)
                should_compute_metrics = True  # Scheduling is owned by SimulationWorker.

                # Compute error metrics (only on frames matching frame skip)
                if solver.iteration > 0 and should_compute_metrics:
                    u_prev_np = np.array(solver.u_prev)
                    v_prev_np = np.array(solver.v_prev)

                    # For MAC grid, interpolate previous velocities to cell centers (current already interpolated above)
                    if grid_type == 'mac':
                        # u is staggered in x (nx+1, ny), interpolate in x: 0.5 * (u[:-1, :] + u[1:, :])
                        u_prev_np = 0.5 * (u_prev_np[:-1, :] + u_prev_np[1:, :])
                        # v is staggered in y (nx, ny+1), interpolate in y: 0.5 * (v[:, :-1] + v[:, 1:])
                        v_prev_np = 0.5 * (v_prev_np[:, :-1] + v_prev_np[:, 1:])

                    delta_u = u_np - u_prev_np
                    delta_v = v_np - v_prev_np

                    dx = float(solver.grid.dx)
                    dy = float(solver.grid.dy)

                    l2_delta_u = np.sqrt(np.sum(delta_u**2) * dx * dy)
                    l2_delta_v = np.sqrt(np.sum(delta_v**2) * dx * dy)
                    l2_delta_total = np.sqrt(l2_delta_u**2 + l2_delta_v**2)

                    max_delta_u = np.max(np.abs(delta_u))
                    max_delta_v = np.max(np.abs(delta_v))
                    max_delta_total = np.maximum(max_delta_u, max_delta_v)

                    # Calculate velocity change magnitude
                    if solver.sim_params.grid_type == 'mac':
                        # For MAC grid, skip delta_mag computation due to shape issues
                        delta_mag = np.zeros((solver.grid.nx, solver.grid.ny))
                    else:
                        delta_mag = np.sqrt(delta_u**2 + delta_v**2)

                    u_rms = np.sqrt(np.sum(u_np**2) * dx * dy / (solver.grid.nx * solver.grid.ny))
                    v_rms = np.sqrt(np.sum(v_np**2) * dx * dy / (solver.grid.nx * solver.grid.ny))
                    vel_rms = np.sqrt(u_rms**2 + v_rms**2) + 1e-8

                    rel_delta = l2_delta_total / (vel_rms * np.sqrt(float(solver.grid.lx) * float(solver.grid.ly)))

                    # Compute divergence only in pure fluid region (mask > 0.99) to exclude IBM transition zone
                    if solver.sim_params.grid_type == 'mac':
                        # Use proper staggered divergence for MAC grid with original staggered arrays
                        from solver.operators_mac import divergence_nonperiodic_staggered
                        div = divergence_nonperiodic_staggered(u_np_staggered, v_np_staggered, dx, dy)
                    else:
                        # Use collocated divergence for collocated grid
                        div_x = np.gradient(u_np, dx, axis=0)
                        div_y = np.gradient(v_np, dy, axis=1)
                        div = div_x + div_y
                    fluid_mask = (mask_np > 0.99)  # Changed from 0.5 to 0.99 to exclude transition zone
                    div_fluid = div * fluid_mask
                    div_rms = np.sqrt(np.sum(div_fluid**2) / (np.sum(fluid_mask) + 1e-8))
                    l2_div = np.sqrt(np.sum(div_fluid**2) * dx * dy)

                    error_metrics = {
                        'l2_change': float(l2_delta_total),
                        'rms_change': float(l2_delta_total / np.sqrt(solver.grid.nx * solver.grid.ny)),
                        'max_change': float(max_delta_total),
                        'change_99p': float(np.percentile(delta_mag, 99)),
                        'rel_change': float(rel_delta),
                        'l2_change_u': float(l2_delta_u),
                        'l2_change_v': float(l2_delta_v),
                        'rms_divergence': float(div_rms),  # Now stores RMS
                        'l2_divergence': float(l2_div),
                        'iteration': iteration
                    }
                else:
                    error_metrics = {
                        'l2_change': 0.0,
                        'rms_change': 0.0,
                        'max_change': 0.0,
                        'change_99p': 0.0,
                        'rel_change': 0.0,
                        'l2_change_u': 0.0,
                        'l2_change_v': 0.0,
                        'rms_divergence': 0.0,  # Now stores RMS divergence
                        'l2_divergence': 0.0,
                        'iteration': iteration
                    }

                # Compute airfoil metrics if enabled and frame skip allows
                airfoil_metrics = None
                self.frame_count += 1
                if solver.compute_airfoil_metrics and solver.sim_params.flow_type == 'von_karman' and should_compute_metrics:
                    try:
                        X_np = np.array(solver.grid.X)
                        Y_np = np.array(solver.grid.Y)
                        # u_np and v_np are already interpolated to cell centers for MAC grid above

                        # Compute vorticity for circulation-based force calculation
                        from solver.operators import vorticity, vorticity_nonperiodic
                        grid_type = getattr(solver.sim_params, 'grid_type', 'collocated')
                        if grid_type == 'mac':
                            from solver.operators_mac import vorticity_staggered, vorticity_nonperiodic_staggered
                            if solver.sim_params.flow_type == 'von_karman' or solver.sim_params.flow_type == 'lid_driven_cavity':
                                w_np = np.array(vorticity_nonperiodic_staggered(u, v, dx, dy))
                            else:
                                w_np = np.array(vorticity_staggered(u, v, dx, dy))
                        else:
                            if solver.sim_params.flow_type == 'von_karman' or solver.sim_params.flow_type == 'lid_driven_cavity':
                                w_np = np.array(vorticity_nonperiodic(u_np, v_np, dx, dy))
                            else:
                                w_np = np.array(vorticity(u_np, v_np, dx, dy))

                        stag_x = find_stagnation_point(u_np, v_np, mask_np, pressure_np, X_np, dx)
                        sep_x = find_separation_point(u_np, v_np, mask_np, X_np, dx, dy)

                        chord_length = getattr(solver.sim_params, 'naca_chord', 2.0)
                        airfoil_x = getattr(solver.sim_params, 'naca_x', 5.0)
                        airfoil_y = getattr(solver.sim_params, 'naca_y', 2.5)

                        # Use circulation-based force calculation (IBM-appropriate)
                        cl, cd = compute_forces_ibm(u_np, v_np, w_np, X_np, Y_np, mask_np,
                                                  dx, dy, solver.flow.U_inf,
                                                  chord_length, airfoil_x, airfoil_y,
                                                  solver.grid.lx,
                                                  grid_type=grid_type)

                        rho = 1.0
                        surface = get_airfoil_surface_mask(mask_np, dx, threshold=0.1)
                        p_inf = 0.0
                        q_inf = 0.5 * rho * solver.flow.U_inf**2
                        cp = (pressure_np - p_inf) / q_inf
                        cp_surface = np.where(surface, cp, np.inf)
                        cp_min = float(np.min(cp_surface))

                        airfoil_x = getattr(solver.sim_params, 'naca_x', 2.5)
                        wake_x = airfoil_x + chord_length
                        wake_x_idx = int(wake_x / dx)
                        wake_deficit = 0.0
                        if 0 <= wake_x_idx < solver.grid.nx:
                            u_wake = u_np[wake_x_idx, :]
                            wake_deficit = float(solver.flow.U_inf - np.mean(u_wake[mask_np[wake_x_idx, :] > 0.5]))

                        airfoil_metrics = {
                            'CL': cl,
                            'CD': cd,
                            'stagnation_x': float(stag_x),
                            'separation_x': float(sep_x),
                            'Cp_min': cp_min,
                            'wake_deficit': wake_deficit,
                            'strouhal': 0.0,  # Will be updated when stability is detected
                            'iteration': iteration
                        }
                    except Exception as e:
                        logger.error(f"Error computing airfoil metrics: {e}")
                        airfoil_metrics = None

                # Only emit metrics signal when we actually computed metrics (not on skipped frames)
                if should_compute_metrics:
                    metrics_data = {
                        'error_metrics': error_metrics,
                        'airfoil_metrics': airfoil_metrics,
                        'time': getattr(solver, 'simulated_time', iteration * solver.dt),
                        'dt': solver.dt,
                        'source': self
                    }
                    if self.running and not self.paused:
                        self.metrics_ready.emit(metrics_data)

            except Exception as e:
                logger.error(f"Error in metrics computation: {e}")
                import traceback
                traceback.print_exc()


class SimulationWorker(QObject):
    """Separate thread for simulation computation using Python threading"""
    failed = pyqtSignal(object, str)
    angle_changed = pyqtSignal(float)
    data_ready = pyqtSignal(object)  # Signal when new data is ready
    fps_update = pyqtSignal(int)     # Signal for FPS updates
    profiling_update = pyqtSignal(float, float, float, float)  # Signal for profiling data (solver_ms, interp_ms, total_ms, sim_fps)

    def __init__(self, solver, control_panel=None, info_panel=None, metrics_worker=None, flow_viz=None):
        super().__init__()
        self.solver = solver
        self.control_panel = control_panel
        self.info_panel = info_panel
        self.metrics_worker = metrics_worker
        self.flow_viz = flow_viz
        self.running = False
        self.paused = False
        self.data_queue = queue.Queue(maxsize=2)  # Buffer of 2 frames
        self.simulation_speed = 1.0  # Simulation speed multiplier
        self.thread = None  # Python thread instead of Qt thread
        
        # Frame skipping
        self.simulation_step_counter = 0
        self.update_every = 1
        
        # FPS tracking for simulation
        self.sim_fps_counter = 0
        self.last_sim_fps_time = time.time()
        
        self.shared_buffers = {}
        self.pending_frame = False
        self.step_lock = threading.RLock()
        self.active_wall_time = 0.0
        self.initial_sim_time = getattr(solver, 'simulated_time', solver.iteration * solver.dt)
        self.sample_ui()
        self.settings_timer = QTimer(self)
        self.settings_timer.timeout.connect(self.sample_ui)
        self.angle_changed.connect(self.show_angle, Qt.ConnectionType.QueuedConnection)

    def sample_ui(self):
        """Only called on the Qt thread; publish an immutable settings tuple."""
        cp, ip, fv = self.control_panel, self.info_panel, self.flow_viz
        diagnostics = ip.diagnostics_checkbox.isChecked() if ip else False
        div = bool(fv and hasattr(fv, 'div_plot') and fv.div_plot.isVisible())
        skip = max(1, int(ip.metrics_frame_skip_input.value())) if ip and hasattr(ip, 'metrics_frame_skip_input') else 100
        dynamic = bool(cp and hasattr(cp, 'dynamic_airfoil_checkbox') and cp.dynamic_airfoil_checkbox.value() == 1)
        motion = (cp.min_aoa_spinbox.value(), cp.max_aoa_spinbox.value(),
                  cp.aoa_increment_spinbox.value(), cp.steps_per_increment_slider.value()) if dynamic else (0., 0., 0., 1)
        pressure = bool(fv and hasattr(fv, 'pressure_plot') and fv.pressure_plot.isVisible())
        self.ui_settings = (diagnostics, div, skip, dynamic, motion, pressure)

    @pyqtSlot(float)
    def show_angle(self, angle):
        if not self.running:
            return
        for name, value in [('angle_spinbox', angle), ('angle_slider', int(angle * 10))]:
            widget = getattr(self.control_panel, name, None)
            if widget is not None:
                widget.blockSignals(True)
                widget.setValue(value)
                widget.blockSignals(False)

    def start(self):
        """Start the simulation thread"""
        try:
            if self.thread is None or not self.thread.is_alive():
                self.running = True
                self.sample_ui()
                self.settings_timer.start(100)
                self.thread = threading.Thread(target=self.run_simulation, daemon=True)
                self.thread.start()
                logger.info("Simulation thread started")
        except Exception as e:
            logger.error(f"Failed to start simulation thread: {e}")
            import traceback
            traceback.print_exc()
            self.running = False
    
    def run_simulation(self):
        """Main simulation loop in separate thread"""
        step_count = 0
        logger.info("Starting simulation loop...")

        # Track iteration rate directly
        iteration_start_time = time.time()

        # ------------------------------------------------------
        # VWT VIEWPORT DECOUPLING
        # CFD runs as fast as possible; GUI refresh is throttled
        # independently.
        # ------------------------------------------------------
        last_viewport_emit_time = 0.0
        last_metrics_iteration = -1000000
        last_metrics_time = 0.0
        batch_size = 1
        viewport_target_fps = 30.0
        viewport_emit_interval = 1.0 / viewport_target_fps

        # Dynamic airfoil motion state
        dynamic_aoa_enabled = False
        dynamic_aoa_direction = 1  # 1 for increasing, -1 for decreasing
        dynamic_aoa_current = 0.0
        dynamic_aoa_step_counter = 0

        while self.running:
            try:
                # If paused, wait and continue
                if self.paused:
                    time.sleep(0.01)
                    continue

                step_count += 1
                
                # Check if we should stop (more responsive)
                if not self.running:
                    logger.info("Simulation stop requested")
                    break
                
                # Run simulation step with coefficient computation for airfoil metrics
                cycle_start = time.perf_counter()
                t_solver_start = time.time()
                try:
                    # Get diagnostics setting from GUI
                    compute_diagnostics, compute_div, metrics_skip, is_dynamic, motion, compute_pressure = self.ui_settings
                    
                    # Always pass compute_diagnostics=False to avoid blocking - metrics computed in separate thread
                    # Only compute divergence if the divergence plot is visible

                    # Bounded device work independent of numerical dt and render cadence.
                    cfd_substeps = batch_size if hasattr(self.solver, 'advance_steps') else 1
                    with self.step_lock:
                        if self.paused or not self.running:
                            continue
                        if hasattr(self.solver, 'advance_steps'):
                            self.solver.advance_steps(cfd_substeps)
                            # Bound asynchronous dispatch and measure completed numerical work.
                            jax.block_until_ready((self.solver.u, self.solver.v))
                        else:
                            u, v, vort, div = self.solver.step_for_visualization(
                                compute_divergence=compute_div, compute_drag_lift=False,
                                compute_diagnostics=False)
                        self.active_wall_time += time.perf_counter() - cycle_start
                except Exception as step_error:
                    logger.error(f"Simulation step failed: {step_error}")
                    import traceback
                    traceback.print_exc()
                    
                    self.failed.emit(self, str(step_error))
                    break
                t_solver_end = time.time()
                # Fit a batch into about 25 ms when the hardware permits it.
                batch_size = max(1, min(32, int(.025 * cfd_substeps / max(t_solver_end - t_solver_start, 1e-6))))
                
                # --------------------------------------------------
                # METRICS THROTTLE
                #
                # Metrics require GPU -> CPU synchronization and must
                # not run on every CFD cycle.
                # --------------------------------------------------
                if (
                    (compute_diagnostics or self.solver.compute_airfoil_metrics)
                    and self.metrics_worker
                    and (
                        self.solver.iteration - last_metrics_iteration >= metrics_skip
                        and time.monotonic() - last_metrics_time >= 0.2
                    )
                ):
                    last_metrics_iteration = self.solver.iteration
                    last_metrics_time = time.monotonic()
                    self.metrics_worker.enqueue_data(
                        self.solver.u,
                        self.solver.v,
                        self.solver.current_pressure,
                        self.solver.mask,
                        self.solver.iteration
                    )
                
                # --------------------------------------------------
                # VIEWPORT UPDATE THROTTLE
                #
                # Do NOT force JAX -> NumPy transfers every CFD cycle.
                # Only prepare display data when the viewport is due.
                # --------------------------------------------------
                now_viewport = time.time()

                should_emit_viewport = (
                    (now_viewport - last_viewport_emit_time)
                    >= viewport_emit_interval
                    and not self.pending_frame
                )

                if should_emit_viewport:
                    if hasattr(self.solver, 'visualization_fields'):
                        u, v, vort, div = self.solver.visualization_fields(compute_divergence=compute_div)

                    # Get divergence from solver history.
                    if (
                        hasattr(self.solver, 'history')
                        and 'rms_divergence' in self.solver.history
                        and self.solver.history['rms_divergence']
                    ):
                        div_rms = (
                            self.solver.history[
                                'rms_divergence'
                            ][-1]
                        )
                    else:
                        div_rms = 0.0

                    # ----------------------------------------------
                    # Prepare visualization fields only now.
                    # ----------------------------------------------
                    t_interp_start = time.time()

                    grid_type = getattr(
                        self.solver.sim_params,
                        'grid_type',
                        'collocated'
                    )

                    solver_type = getattr(
                        self.solver.sim_params,
                        'solver_type',
                        'navier_stokes'
                    )

                    if (
                        solver_type == 'lattice_boltzmann'
                        or grid_type == 'collocated'
                    ):
                        u_display = u
                        v_display = v
                        vort_display = vort
                        vel_mag_display = jnp.sqrt(
                            u * u + v * v
                        )

                    elif grid_type == 'mac':

                        u_center = 0.5 * (
                            u[1:, :] + u[:-1, :]
                        )

                        v_center = 0.5 * (
                            v[:, 1:] + v[:, :-1]
                        )

                        u_display = u_center
                        v_display = v_center

                        vel_mag_display = jnp.sqrt(
                            u_center * u_center
                            + v_center * v_center
                        )

                        if vort.shape == (
                            self.solver.grid.nx + 1,
                            self.solver.grid.ny
                        ):
                            vort_display = 0.5 * (
                                vort[1:, :] + vort[:-1, :]
                            )

                        elif vort.shape == (
                            self.solver.grid.nx,
                            self.solver.grid.ny + 1
                        ):
                            vort_display = 0.5 * (
                                vort[:, 1:] + vort[:, :-1]
                            )

                        else:
                            vort_display = vort

                    else:
                        u_display = u
                        v_display = v
                        vort_display = vort
                        vel_mag_display = jnp.sqrt(
                            u * u + v * v
                        )

                    t_interp_end = time.time()

                    # ----------------------------------------------
                    # GPU -> CPU transfer only for displayed frames.
                    # ----------------------------------------------
                    t_queue_start = time.time()

                    scalar_field = getattr(self.solver, 'c', None)
                    if hasattr(self.solver, 'lbm_params') and getattr(self.solver.lbm_params, 'enable_thermal', False):
                        scalar_field = getattr(self.solver, 'T', None)
                    fields = jax.device_get({
                        'u': u_display, 'v': v_display, 'vort': vort_display,
                        'vel_mag': vel_mag_display, 'div': div, 'scalar': scalar_field,
                        'pressure': self.solver.current_pressure if compute_pressure else None,
                    })
                    data = {key: np.asarray(value, dtype=np.float32) if value is not None else None
                            for key, value in fields.items()}
                    data.update({
                        'time': getattr(self.solver, 'simulated_time', self.solver.iteration * self.solver.dt),
                        'dt': self.solver.dt, 'source': self, 'rtf': 0.0,
                        'iteration': self.solver.iteration, 'rms_divergence': div_rms,
                    })

                    # RTF includes simulation, field preparation and transfers; excludes pause.
                    wall = self.active_wall_time + time.perf_counter() - cycle_start - (t_solver_end - t_solver_start)
                    data['rtf'] = (data['time'] - self.initial_sim_time) / max(wall, 1e-12)
                    t_queue_end = time.time()

                    t_signal_start = time.time()

                    if self.running:
                        try:
                            self.pending_frame = True
                            self.data_ready.emit(data)
                        except Exception as emit_error:
                            logger.error(
                                "Signal emission failed: "
                                f"{emit_error}"
                            )
                            import traceback
                            traceback.print_exc()

                    t_signal_end = time.time()

                    last_viewport_emit_time = now_viewport

                else:
                    # Profiling variables still need valid values.
                    t_interp_start = t_solver_end
                    t_interp_end = t_solver_end
                    t_queue_start = t_solver_end
                    t_queue_end = t_solver_end
                    t_signal_start = t_solver_end
                    t_signal_end = t_solver_end

                # Dynamic airfoil motion logic
                if self.control_panel:
                    # Check if dynamic mode is enabled and obstacle is NACA

                    is_naca = getattr(self.solver.sim_params, 'obstacle_type', '') == 'naca_airfoil'

                    if is_dynamic and is_naca:
                        # Get parameters from UI
                        min_aoa, max_aoa, aoa_increment, steps_per_increment = motion

                        # Initialize on first iteration
                        if not dynamic_aoa_enabled:
                            dynamic_aoa_enabled = True
                            dynamic_aoa_current = min_aoa
                            dynamic_aoa_direction = 1
                            dynamic_aoa_step_counter = 0
                            # Update solver with initial AoA
                            if hasattr(self.solver, 'update_naca_angle'):
                                self.solver.update_naca_angle(dynamic_aoa_current, recompute=True)

                        # Increment step counter
                        dynamic_aoa_step_counter += cfd_substeps

                        # Check if we should update AoA
                        if dynamic_aoa_step_counter >= steps_per_increment:
                            dynamic_aoa_step_counter = 0

                            # Update AoA
                            dynamic_aoa_current += dynamic_aoa_direction * aoa_increment

                            # Check bounds and reverse direction if needed
                            if dynamic_aoa_current >= max_aoa:
                                dynamic_aoa_current = max_aoa
                                dynamic_aoa_direction = -1
                            elif dynamic_aoa_current <= min_aoa:
                                dynamic_aoa_current = min_aoa
                                dynamic_aoa_direction = 1

                            # Update solver with new AoA
                            if hasattr(self.solver, 'update_naca_angle'):
                                self.solver.update_naca_angle(dynamic_aoa_current, recompute=True)

                                # Clear JIT cache and recompile to pick up new mask
                                if hasattr(self.solver, '_jit_cache'):
                                    self.solver._jit_cache.clear()
                                if hasattr(self.solver, '_step_jit'):
                                    delattr(self.solver, '_step_jit')
                                self.solver._step_jit = self.solver.get_step_jit()

                                self.angle_changed.emit(dynamic_aoa_current)
                    else:
                        # Dynamic mode disabled, reset state
                        dynamic_aoa_enabled = False

                # PNG mask spin logic (for LBM with custom masks)
                if hasattr(self.solver, 'lbm_params') and self.solver.lbm_params.enable_spin:
                    if hasattr(self.solver, 'update_spin'):
                        self.solver.update_spin()

                self.active_wall_time += max(0.0, time.perf_counter() - cycle_start - (t_solver_end - t_solver_start))
                # Simulation FPS counter
                self.sim_fps_counter += cfd_substeps
                if time.time() - self.last_sim_fps_time > 1.0:
                    try:
                        step_rate = self.sim_fps_counter / max(time.time() - self.last_sim_fps_time, 1e-9)
                        self.fps_update.emit(round(step_rate))
                        # Emit profiling data (every second)
                        solver_ms = (t_solver_end - t_solver_start) * 1000
                        interp_ms = (t_interp_end - t_interp_start) * 1000
                        total_ms = (t_signal_end - t_solver_start) * 1000
                        self.profiling_update.emit(solver_ms, interp_ms, total_ms, step_rate)
                    except Exception as fps_error:
                        logger.error(f"FPS update failed: {fps_error}")
                    self.sim_fps_counter = 0
                    self.last_sim_fps_time = time.time()
                
            except Exception as e:
                if not self.running:  # Error during shutdown is OK
                    break
                self.failed.emit(self, str(e))
                logger.critical(f"Simulation loop crashed: {e}")
                import traceback
                traceback.print_exc()
                break  # Exit the loop on fatal error
        
        self.running = False
        logger.info("Simulation thread stopped")
    
    def pause(self):
        """Pause the simulation"""
        self.paused = True
        with self.step_lock:
            pass  # Return only once the in-flight numerical batch has completed.

    def resume(self):
        """Resume the simulation"""
        self.paused = False

    def recreate_shared_buffers(self, new_nx, new_ny):
        """Recreate shared memory buffers for new grid dimensions"""
        # Frames own their NumPy arrays; no shared-memory allocation is needed.
        self.shared_buffers = {}

    def stop_simulation(self):
        """Stop the simulation thread with robust cleanup"""
        self.settings_timer.stop()
        self.running = False
        self.paused = False  # Ensure paused state is cleared
        
        # Wait for thread to stop with longer timeout for robustness
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)  # Wait up to 5 seconds
            if self.thread.is_alive():
                logger.warning("Simulation thread did not stop gracefully after 5 seconds")
                raise RuntimeError("Simulation worker still stopping; refusing concurrent reset/restart")

        # Clean up shared memory
        if hasattr(self, 'shared_buffers'):
            for buffer in self.shared_buffers.values():
                try:
                    buffer.cleanup()
                except Exception as e:
                    logger.warning(f"Buffer cleanup error: {e}")
    
    def pause_simulation(self):
        """Pause the simulation without stopping the thread"""
        self.paused = True
        logger.info("Simulation paused")
    
    def resume_simulation(self):
        """Resume the simulation from paused state"""
        self.paused = False
        logger.info("Simulation resumed")


class SimulationController:
    """Controls simulation execution and data flow"""
    
    def __init__(self, solver, control_panel=None, info_panel=None, flow_viz=None):
        self.solver = solver
        self.control_panel = control_panel
        self.info_panel = info_panel
        self.flow_viz = flow_viz
        self.simulation_worker = None
        self.metrics_worker = None
        self.latest_data = None
        self.latest_metrics = None
        self.callbacks = None  # Store callbacks for reconnecting signals
        self.running = False  # Running state for compatibility checks
        self.backend = "cfd"
        self.physicsnemo_case = {"geometry_id": "cylinder-1", "U_inf": 0.75, "Re": 200.0}

        # Frame skipping controls (kept for compatibility with existing code)
        self.simulation_step_counter = 0
        self.update_every = 1  # Default to 1 (no skipping)
        self.should_update_visualization = False

        # Performance tracking
        self.frame_counter = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        
    def update_grid_size(self, new_nx, new_ny):
        """Update simulation worker for new grid size - COMPLETE RESTART"""
        # Force complete shutdown
        self.stop_simulation()
        
        # Clear all references to force garbage collection
        if self.simulation_worker:
            try:
                if hasattr(self.simulation_worker, 'shared_buffers'):
                    for buffer in self.simulation_worker.shared_buffers.values():
                        buffer.cleanup()
            except Exception as cleanup_error:
                logger.warning(f"Buffer cleanup error: {cleanup_error}")
            finally:
                self.simulation_worker = None
        
        # Clear latest data to prevent stale references
        self.latest_data = None
        
        # Force garbage collection
        import gc
        gc.collect()

    def start_simulation(self, callbacks):
        """Start simulation in separate thread"""
        # Store callbacks for reconnecting signals when metrics worker is restarted
        self.callbacks = callbacks
        
        try:
            self.stop_simulation()
            self.simulation_worker = None
            self.latest_data = None
            self.latest_metrics = None
            if self.backend == "physicsnemo":
                from viewer.physicsnemo_backend import PhysicsNeMoWorker
                self.metrics_worker = None
                self.simulation_worker = PhysicsNeMoWorker(self.solver, self.physicsnemo_case.copy())
            else:
                self.metrics_worker = MetricsWorker(self.solver)
                self.metrics_worker.start()
                self.simulation_worker = SimulationWorker(self.solver, self.control_panel, self.info_panel, self.metrics_worker, self.flow_viz)

            # Connect signals
            if 'data_ready' in callbacks:
                self.simulation_worker.data_ready.connect(
                    callbacks['data_ready'], Qt.ConnectionType.QueuedConnection
                )
            if 'fps_update' in callbacks:
                self.simulation_worker.fps_update.connect(
                    callbacks['fps_update'], Qt.ConnectionType.QueuedConnection
                )
            if 'profiling_update' in callbacks:
                self.simulation_worker.profiling_update.connect(
                    callbacks['profiling_update'], Qt.ConnectionType.QueuedConnection
                )
            if 'metrics_ready' in callbacks and self.metrics_worker:
                self.metrics_worker.metrics_ready.connect(
                    callbacks['metrics_ready'], Qt.ConnectionType.QueuedConnection
                )

            if 'failed' in callbacks:
                self.simulation_worker.failed.connect(callbacks['failed'], Qt.ConnectionType.QueuedConnection)
            # Start the thread
            self.simulation_worker.start()
            self.running = True

            logger.info("Simulation started in separate thread")

        except Exception as e:
            logger.error(f"Error starting simulation: {e}")
            import traceback
            traceback.print_exc()
            raise

    def configure_backend(self, backend, case=None):
        """Select the worker implementation without changing the CFD solver."""
        if backend not in ("cfd", "physicsnemo"):
            raise ValueError(f"Unknown solver backend: {backend}")
        if self.running or (self.simulation_worker and self.simulation_worker.running):
            self.stop_simulation()
        self.backend = backend
        if case is not None:
            self.physicsnemo_case = dict(case)
    
    def full_reset(self):
        """Completely reset the simulation controller for parameter changes."""

        self.stop_simulation()
        self.simulation_worker = None
        self.latest_data = None
        self.latest_metrics = None
        self.simulation_step_counter = 0
        self.should_update_visualization = False

    def start_metrics(self):
        """Ensure exactly one MetricsWorker is running."""

        created = False

        if self.metrics_worker is None:
            self.metrics_worker = MetricsWorker(self.solver)
            created = True
        else:
            self.metrics_worker.solver = self.solver

        worker_alive = (
            self.metrics_worker.running
            and self.metrics_worker.thread is not None
            and self.metrics_worker.thread.is_alive()
        )

        if not worker_alive:
            self.metrics_worker.start()

        if (
            created
            and self.callbacks
            and "metrics_ready" in self.callbacks
        ):
            self.metrics_worker.metrics_ready.connect(
                self.callbacks["metrics_ready"],
                Qt.ConnectionType.QueuedConnection
            )

        if self.simulation_worker:
            self.simulation_worker.metrics_worker = (
                self.metrics_worker
            )

    def stop_metrics(self):
        """Stop and release MetricsWorker."""

        if self.metrics_worker is None:
            return

        worker = self.metrics_worker
        worker.stop()
        self.metrics_worker = None

        if self.simulation_worker:
            self.simulation_worker.metrics_worker = None

    def stop_simulation(self):
        """Stop simulation and metrics workers safely."""

        self.running = False

        if self.simulation_worker:
            self.simulation_worker.stop_simulation()

        self.stop_metrics()
        self.latest_data = None
        self.latest_metrics = None
        self.should_update_visualization = False

        logger.info("Simulation stopped")

    def pause_simulation(self):
        """Pause simulation without stopping the thread"""
        if self.simulation_worker:
            self.simulation_worker.pause()
        if self.metrics_worker:
            self.metrics_worker.pause()
    
    def resume_simulation(self):
        """Resume simulation from paused state"""
        if self.simulation_worker:
            self.simulation_worker.resume()
        if self.metrics_worker:
            self.metrics_worker.resume()
    
    def on_simulation_data_ready(self, data):
        """Handle new data from simulation thread"""
        # Store the latest data
        self.latest_data = data
        
        # Increment simulation step counter for frame skipping
        self.simulation_step_counter += 1
        
        # Frame skipping: only update visualization every N simulation steps
        if self.simulation_step_counter % self.update_every == 0:
            # This is a frame that should be visualized
            self.should_update_visualization = True
        else:
            # Skip this frame
            self.should_update_visualization = False
    
    def set_frame_skip(self, update_every):
        """Set frame skip setting"""
        self.update_every = update_every
    
    def get_latest_data(self):
        """Get the latest simulation data"""
        return self.latest_data
    
    def should_update(self):
        """Check if visualization should be updated"""
        return self.should_update_visualization
    
    def reset_update_flag(self):
        """Reset the visualization update flag"""
        self.should_update_visualization = False


class RecordingManager:
    """Manages video recording functionality"""
    
    def __init__(self):
        self.is_recording = False
        self.recorded_frames = []
    
    def toggle_recording(self):
        """Toggle video recording"""
        if not self.is_recording:
            self.is_recording = True
            self.recorded_frames = []
            logger.info("Started recording video...")
            return "Stop Recording"
        else:
            self.is_recording = False
            logger.info(f"Stopped recording. Captured {len(self.recorded_frames)} frames.")
            return "Record Video"
    
    def capture_frame(self, frame_data):
        """Capture a frame for recording"""
        if self.is_recording:
            try:
                # Normalize to 0-255 for video
                frame_norm = ((frame_data - frame_data.min()) / 
                             (frame_data.max() - frame_data.min()) * 255).astype(np.uint8)
                self.recorded_frames.append(frame_norm)
            except:
                pass  # Skip frame if capture fails
    
    def save_video(self, parent_widget=None):
        """Save recorded frames as video"""
        if not self.recorded_frames:
            logger.warning("No frames to save!")
            return
            
        try:
            import imageio
            from PyQt6.QtWidgets import QFileDialog
            
            filename, _ = QFileDialog.getSaveFileName(
                parent_widget, "Save Video", "flow_simulation.mp4", "Video Files (*.mp4 *.avi)"
            )
            if filename:
                logger.info(f"Saving video with {len(self.recorded_frames)} frames...")
                imageio.mimsave(filename, self.recorded_frames, fps=30)
                logger.info(f"Video saved as {filename}")
        except ImportError:
            logger.error("imageio not installed. Install with: pip install imageio")
        except Exception as e:
            logger.error(f"Error saving video: {e}")
    
    def has_frames(self):
        """Check if there are frames to save"""
        return len(self.recorded_frames) > 0
    
    def get_frame_count(self):
        """Get number of recorded frames"""
        return len(self.recorded_frames)


class DataExporter:
    """Handles data export functionality"""
    
    @staticmethod
    def export_simulation_data(solver, history_data=None):
        """Export simulation data to files"""
        try:
            import datetime
            import json
            
            # Create timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Get current data
            u_np = np.array(solver.u)
            v_np = np.array(solver.v)
            
            # Get vorticity from cache if available, otherwise compute
            if hasattr(solver, 'current_vorticity') and solver.current_vorticity is not None:
                vort_np = np.array(solver.current_vorticity)
            else:
                vort_np = np.zeros_like(u_np)  # Fallback
            
            # Export to CSV files
            np.savetxt(f'velocity_u_{timestamp}.csv', u_np, delimiter=',')
            np.savetxt(f'velocity_v_{timestamp}.csv', v_np, delimiter=',')
            np.savetxt(f'vorticity_{timestamp}.csv', vort_np, delimiter=',')
            
            # Export history data if provided
            if history_data:
                DataExporter._export_history_data(history_data, timestamp)
            
            # Export grid information
            DataExporter._export_grid_info(solver, timestamp)
            
            logger.info(f"Data exported successfully with timestamp {timestamp}")
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
    
    @staticmethod
    def _export_history_data(history_data, timestamp):
        """Export history data to CSV"""
        try:
            time_data = history_data.get('time', [])
            enst_data = history_data.get('enstrophy', [])
            drag_data = history_data.get('drag', [])
            lift_data = history_data.get('lift', [])
            
            if time_data and len(time_data) > 0:
                history_array = np.column_stack([
                    time_data[:len(time_data)],
                    enst_data[:len(time_data)], 
                    drag_data[:len(time_data)], 
                    lift_data[:len(time_data)]
                ])
                np.savetxt(f'history_{timestamp}.csv', history_array, delimiter=',',
                          header='time,enstrophy,drag,lift')
        except Exception as e:
            logger.error(f"Error exporting history data: {e}")
    
    @staticmethod
    def _export_grid_info(solver, timestamp):
        """Export grid information to JSON"""
        try:
            import json
            
            grid_info = {
                'nx': solver.grid.nx,
                'ny': solver.grid.ny,
                'lx': solver.grid.lx,
                'ly': solver.grid.ly,
                'dx': solver.grid.dx,
                'dy': solver.grid.dy,
                'flow_type': solver.sim_params.flow_type,
                'Re': solver.flow.Re,
                'U_inf': solver.flow.U_inf,
                'nu': solver.flow.nu,
                'dt': solver.dt,
                'iteration': solver.iteration
            }
            
            with open(f'grid_info_{timestamp}.json', 'w') as f:
                json.dump(grid_info, f, indent=2)
        except Exception as e:
            logger.error(f"Error exporting grid info: {e}")
