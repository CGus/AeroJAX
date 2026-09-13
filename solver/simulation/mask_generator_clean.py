import jax.numpy as jnp

def _compute_mask(self):
    """Compute the obstacle mask based on simulation parameters."""
    
    if hasattr(self.sim_params, 'obstacle_type') and self.sim_params.obstacle_type == 'naca_airfoil':
        from obstacles.naca_airfoils import NACAParams, create_naca_mask, parse_naca_4digit, parse_naca_5digit
        
        # Parse NACA designation
        naca_str = self.sim_params.naca_airfoil.upper().replace('NACA', '').strip()
        if len(naca_str) == 4:
            m, p, t = parse_naca_4digit(naca_str)
            airfoil_type = '4-digit'
        elif len(naca_str) == 5:
            cl, p, m, t = parse_naca_5digit(naca_str)
            airfoil_type = '5-digit'
        else:
            raise ValueError(f"Unsupported NACA designation: {self.sim_params.naca_airfoil}")
        
        naca_params = NACAParams(
            airfoil_type=airfoil_type,
            designation=self.sim_params.naca_airfoil,
            chord_length=self.sim_params.naca_chord,
            angle_of_attack=self.sim_params.naca_angle,
            position_x=self.sim_params.naca_x,
            position_y=self.sim_params.naca_y
        )
        # SHARP mask: use simple threshold on SDF (no sigmoid smoothing)
        # Use user's epsilon setting from slider (eps = eps_multiplier * dx)
        epsilon = self.sim_params.eps  # User-controlled via GUI slider (now used as threshold)
        # Get SDF from NACA function, then apply SHARP threshold
        from obstacles.naca_airfoils import naca_surface_distance
        if airfoil_type == '4-digit':
            sdf = naca_surface_distance(self.grid.X, self.grid.Y, naca_params.chord_length,
                                       naca_params.angle_of_attack, naca_params.position_x,
                                       naca_params.position_y, m, p, t)
        else:  # 5-digit
            sdf = naca_surface_distance(self.grid.X, self.grid.Y, naca_params.chord_length,
                                       naca_params.angle_of_attack, naca_params.position_x,
                                       naca_params.position_y, cl, p, m, t)
        # SHARP mask: 1 in fluid (sdf > 0), 0 in solid (sdf < 0)
        # Use epsilon as a small threshold to avoid numerical issues at exact boundary
        mask = jnp.where(sdf > -epsilon, 1.0, 0.0)
        return mask
    elif hasattr(self.sim_params, 'obstacle_type') and self.sim_params.obstacle_type == 'cow':
        from obstacles.cow import sdf_cow_side
        # Compute cow position relative to grid bounds
        # Use cow_x and cow_y from sim_params if available, otherwise use defaults
        cow_x = getattr(self.sim_params, 'cow_x', self.grid.lx * 0.25)  # 25% of domain width default
        cow_y = getattr(self.sim_params, 'cow_y', self.grid.ly * 0.35)  # 35% of domain height default
        # Compute scale factor based on grid dimensions relative to reference (20x3.75)
        ref_lx = 20.0
        ref_ly = 3.75
        scale_x = self.grid.lx / ref_lx
        scale_y = self.grid.ly / ref_ly
        cow_scale = (scale_x + scale_y) / 2.0  # Average of x and y scaling
        # SHARP mask: use simple threshold on SDF (no sigmoid smoothing)
        # Use user's epsilon setting from slider (eps = eps_multiplier * dx)
        epsilon = self.sim_params.eps  # User-controlled via GUI slider (now used as threshold)
        sdf = sdf_cow_side(self.grid.X, self.grid.Y, cow_x, cow_y, cow_scale)
        # SHARP mask: 1 in fluid (sdf > 0), 0 in solid (sdf < 0)
        # Use epsilon as a small threshold to avoid numerical issues at exact boundary
        mask = jnp.where(sdf > -epsilon, 1.0, 0.0)
        return mask
    elif hasattr(self.sim_params, 'obstacle_type') and self.sim_params.obstacle_type == 'three_cylinder_array':
        from obstacles.cylinder_array import sdf_three_cylinders
        cylinder_x = getattr(self.sim_params, 'cylinder_x', 5.0)
        cylinder_y = getattr(self.sim_params, 'cylinder_y', self.grid.ly / 2.0)
        cylinder_diameter = getattr(self.sim_params, 'cylinder_diameter', 0.5)
        cylinder_spacing = getattr(self.sim_params, 'cylinder_spacing', 0.5)
        # SHARP mask: use simple threshold on SDF (no sigmoid smoothing)
        # Use user's epsilon setting from slider (eps = eps_multiplier * dx)
        epsilon = self.sim_params.eps  # User-controlled via GUI slider (now used as threshold)
        sdf = sdf_three_cylinders(self.grid.X, self.grid.Y, cylinder_x, cylinder_y, cylinder_diameter, cylinder_spacing)
        # SHARP mask: 1 in fluid (sdf > 0), 0 in solid (sdf < 0)
        # Use epsilon as a small threshold to avoid numerical issues at exact boundary
        mask = jnp.where(sdf > -epsilon, 1.0, 0.0)
        return mask
    elif hasattr(self.sim_params, 'obstacle_type') and self.sim_params.obstacle_type == 'solid_wall':
        # Solid wall: thin vertical wall with configurable position and y-bounds
        X, Y = self.grid.X, self.grid.Y

        # Get actual Y bounds of grid
        y_min = jnp.min(Y)
        y_max = jnp.max(Y)
        y_range = y_max - y_min

        # Wall parameters from sim_params
        wall_x = self.grid.lx * getattr(self.sim_params, 'solid_wall_x', 0.25)
        wall_thickness = self.grid.lx * 0.02  # 2% of domain width

        # Wall y-bounds from sim_params (as percentages of domain)
        wall_y_start = y_min + y_range * getattr(self.sim_params, 'solid_wall_y_bottom', 0.0)
        wall_y_end = y_min + y_range * getattr(self.sim_params, 'solid_wall_y_top', 0.5)

        # Create signed distance function (SDF) for wall
        # Distance to wall centerline in X, clamped in Y
        dx_wall = jnp.abs(X - wall_x) - wall_thickness / 2

        # For Y: distance to wall's Y-range (0 inside range)
        dy_above = Y - wall_y_end  # positive above wall
        dy_below = wall_y_start - Y  # positive below wall
        dy_wall = jnp.maximum(0, jnp.maximum(dy_above, dy_below))  # 0 inside wall height

        # Combined SDF: outside if outside in X OR outside in Y
        sdf = jnp.maximum(dx_wall, dy_wall)

        # SHARP mask: use simple threshold on SDF (no sigmoid smoothing)
        # Use user's epsilon setting from slider (eps = eps_multiplier * dx)
        epsilon = self.sim_params.eps  # User-controlled via GUI slider (now used as threshold)
        # SHARP mask: 1 in fluid (sdf > 0), 0 in solid (sdf < 0)
        # Use epsilon as a small threshold to avoid numerical issues at exact boundary
        mask = jnp.where(sdf > -epsilon, 1.0, 0.0)
        return mask
    elif hasattr(self.sim_params, 'obstacle_type') and self.sim_params.obstacle_type == 'custom':
        from obstacles.freeform_drawer import create_freeform_mask_smooth
        custom_mask = getattr(self.sim_params, 'custom_mask', None)
        if custom_mask is not None:
            # Use user's epsilon setting from slider
            epsilon = self.sim_params.eps
            # Get obstacle center position from sliders
            center_x = getattr(self.sim_params, 'custom_x', self.grid.lx * 0.25)
            center_y = getattr(self.sim_params, 'custom_y', self.grid.ly * 0.5)
            # Scale custom obstacle to fit in domain while preserving aspect ratio
            # Use the smaller dimension to determine scale, so the drawing fits
            mask_height, mask_width = custom_mask.shape
            
            # Calculate scale to fit in domain (use 60% of smaller dimension)
            domain_min_dim = min(self.grid.lx, self.grid.ly)
            scale = domain_min_dim * 0.6
            
            # Use same scale for both dimensions to preserve aspect ratio
            scale_x = scale
            scale_y = scale
            
            # Calculate offset to center the obstacle at the specified position
            # offset is the center position
            offset_x = center_x
            offset_y = center_y
            
            mask = create_freeform_mask_smooth(self.grid.X, self.grid.Y, custom_mask, 
                                              scale_x=scale_x, scale_y=scale_y,
                                              offset_x=offset_x, offset_y=offset_y,
                                              smooth_width=epsilon)
            return mask
        else:
            # Fallback to cylinder if no custom mask - SHARP mask
            X, Y = self.grid.X, self.grid.Y
            phi = jnp.sqrt((X - self.geom.center_x)**2 + (Y - self.geom.center_y)**2) - self.geom.radius
            epsilon = self.sim_params.eps
            # SHARP mask: 1 in fluid (phi > 0), 0 in solid (phi < 0)
            # Use epsilon as a small threshold to avoid numerical issues at exact boundary
            mask = jnp.where(phi > -epsilon, 1.0, 0.0)
            return mask
    elif hasattr(self.sim_params, 'obstacle_type') and self.sim_params.obstacle_type == 'urban_map':
        # Urban map: use precomputed SDF field from sim_params
        sdf_field = getattr(self.sim_params, 'sdf_field', None)
        print(f"DEBUG urban_map: obstacle_type={getattr(self.sim_params, 'obstacle_type', 'NOT_SET')}")
        print(f"DEBUG urban_map: sdf_field is {type(sdf_field)}, shape: {sdf_field.shape if sdf_field is not None else 'None'}")
        
        if sdf_field is None:
            print("ERROR: urban_map selected but no sdf_field set, falling back to cylinder")
            # Fall back to cylinder to prevent crash
            X, Y = self.grid.X, self.grid.Y
            phi = jnp.sqrt((X - self.geom.center_x)**2 + (Y - self.geom.center_y)**2) - self.geom.radius
            epsilon = self.sim_params.eps
            mask = jnp.where(phi > -epsilon, 1.0, 0.0)
            print(f"FALLBACK: Using cylinder mask - min={mask.min():.3f}, max={mask.max():.3f}")
            return mask
        
        # Convert numpy SDF to JAX if needed
        import numpy as np
        if isinstance(sdf_field, np.ndarray):
            sdf_field = jnp.array(sdf_field)
        
        print(f"DEBUG urban_map: SDF min={sdf_field.min():.3f}, max={sdf_field.max():.3f}, mean={sdf_field.mean():.3f}")
        
        epsilon = self.sim_params.eps
        print(f"DEBUG urban_map: epsilon={epsilon:.6f}")
        
        # Check for individual building SDFs to preserve disconnected buildings
        individual_sdfs = getattr(self.sim_params, 'individual_sdfs', [])
        
        if individual_sdfs and len(individual_sdfs) > 0:
            print(f"DEBUG urban_map: Computing mask from {len(individual_sdfs)} individual building SDFs")
            # Start with all fluid (1.0)
            combined_mask = jnp.ones_like(self.grid.X)
            
            for i, ind_sdf in enumerate(individual_sdfs):
                if ind_sdf is None:
                    continue
                # Convert to JAX array if needed
                if isinstance(ind_sdf, np.ndarray):
                    ind_sdf = jnp.array(ind_sdf)
                
                # Use SHARP threshold for each building (not smooth sigmoid)
                # This prevents building gradients from merging
                ind_mask = jnp.where(ind_sdf > -epsilon, 1.0, 0.0)
                
                # Combine with minimum: if ANY building is solid (0), cell is solid
                combined_mask = jnp.minimum(combined_mask, ind_mask)
                
                solid_cells = jnp.sum(ind_mask < 0.5)
                print(f"DEBUG urban_map: Building {i} solid cells: {solid_cells}")
            
            print(f"DEBUG urban_map: Combined mask min={combined_mask.min():.3f}, max={combined_mask.max():.3f}, mean={combined_mask.mean():.3f}")
            return combined_mask
        else:
            # Fallback: use smooth sigmoid on combined SDF
            print(f"DEBUG urban_map: No individual SDFs, using combined SDF with smooth mask")
            from .mask_to_sdf import sdf_to_mask
            mask = sdf_to_mask(sdf_field, eps=epsilon, smooth=True)
            print(f"DEBUG urban_map: Mask min={mask.min():.3f}, max={mask.max():.3f}, mean={mask.mean():.3f}")
            return mask
    elif hasattr(self.sim_params, 'obstacle_type') and self.sim_params.obstacle_type == 'tesla_valve':
        # Tesla-type valvular conduit.
        #
        # mask convention:
        #   1.0 = fluid
        #   0.0 = solid
        #
        # Geometry:
        # - straight main channel
        # - asymmetric smooth bypass per stage
        # - solid island naturally remains between main path and bypass
        #
        # The geometry itself NEVER changes between forward/backward flow.

        X, Y = self.grid.X, self.grid.Y

        num_stages = int(getattr(
            self.sim_params, 'tesla_valve_stages', 3
        ))
        stage_length = float(getattr(
            self.sim_params, 'tesla_valve_stage_length', 2.0
        ))
        main_width = float(getattr(
            self.sim_params, 'tesla_valve_main_width', 0.55
        ))
        branch_width = float(getattr(
            self.sim_params, 'tesla_valve_branch_width', 0.22
        ))
        diagonal_length = float(getattr(
            self.sim_params, 'tesla_valve_diagonal_length', 0.8
        ))

        valve_x = float(getattr(
            self.sim_params, 'tesla_valve_x', self.grid.lx * 0.5
        ))
        valve_y = float(getattr(
            self.sim_params, 'tesla_valve_y', self.grid.ly * 0.5
        ))

        def tube_segment(x1, y1, x2, y2, radius):
            vx = x2 - x1
            vy = y2 - y1
            vv = vx * vx + vy * vy + 1e-12

            t = ((X - x1) * vx + (Y - y1) * vy) / vv
            t = jnp.clip(t, 0.0, 1.0)

            px = x1 + t * vx
            py = y1 + t * vy

            d2 = (X - px) ** 2 + (Y - py) ** 2
            return d2 <= radius ** 2

        def quadratic_bezier_tube(p0, p1, p2, radius, segments=18):
            result = jnp.zeros_like(X, dtype=bool)

            prev_x, prev_y = p0

            for k in range(1, segments + 1):
                t = k / segments
                omt = 1.0 - t

                x = (
                    omt * omt * p0[0]
                    + 2.0 * omt * t * p1[0]
                    + t * t * p2[0]
                )
                y = (
                    omt * omt * p0[1]
                    + 2.0 * omt * t * p1[1]
                    + t * t * p2[1]
                )

                result = result | tube_segment(
                    prev_x, prev_y, x, y, radius
                )

                prev_x, prev_y = x, y

            return result

        # Main passage extends across the whole computational domain.
        fluid = jnp.abs(Y - valve_y) <= main_width * 0.5

        total_length = num_stages * stage_length
        x_start = valve_x - total_length * 0.5

        bypass_radius = max(branch_width * 0.5, 0.055)

        # Height of the bypass above the main conduit.
        bypass_height = max(
            main_width * 1.65,
            diagonal_length
        )

        for i in range(num_stages):
            sx = x_start + i * stage_length

            # Asymmetric connection locations.
            # Short/steep entrance + long/gentle return.
            x_in = sx + stage_length * 0.18
            x_top = sx + stage_length * 0.43
            x_out = sx + stage_length * 0.90

            y_wall = valve_y + main_width * 0.38
            y_top = valve_y + main_width * 0.5 + bypass_height

            # Steep branch leaving main conduit.
            branch_up = quadratic_bezier_tube(
                (x_in, y_wall),
                (
                    sx + stage_length * 0.20,
                    valve_y + bypass_height * 0.95
                ),
                (x_top, y_top),
                bypass_radius,
                segments=14
            )

            # Long curved return downstream.
            branch_return = quadratic_bezier_tube(
                (x_top, y_top),
                (
                    sx + stage_length * 0.78,
                    y_top
                ),
                (x_out, y_wall),
                bypass_radius,
                segments=22
            )

            fluid = fluid | branch_up | branch_return

        self.sdf = None
        mask = jnp.where(fluid, 1.0, 0.0)

        print(
            f"Tesla valve V2 mask: stages={num_stages}, "
            f"L={stage_length:.2f}, main={main_width:.2f}, "
            f"branch={branch_width:.2f}, "
            f"position=({valve_x:.2f}, {valve_y:.2f})"
        )

        return mask
    else:
        # Fallback to cylinder if no custom mask - SHARP mask
        X, Y = self.grid.X, self.grid.Y
        phi = jnp.sqrt((X - self.geom.center_x)**2 + (Y - self.geom.center_y)**2) - self.geom.radius
        epsilon = self.sim_params.eps
        # SHARP mask: 1 in fluid (phi > 0), 0 in solid (phi < 0)
        # Use epsilon as a small threshold to avoid numerical issues at exact boundary
        mask = jnp.where(phi > -epsilon, 1.0, 0.0)
        return mask
