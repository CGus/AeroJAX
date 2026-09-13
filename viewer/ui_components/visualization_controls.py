"""
Visualization settings controls.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QCheckBox, QSlider, QComboBox, QGridLayout, QSizePolicy, QSizePolicy
)
from ..config import ConfigManager
from .collapsible_groupbox import CollapsibleGroupBox


class VisualizationControls(CollapsibleGroupBox):
    """Group for all visualization settings (performance, toggles, colormaps, export)"""

    def __init__(self, parent=None):
        super().__init__("Visualization")
        self.parent_viewer = parent
        self.setup_ui()

    def setup_ui(self):
        """Setup visualization controls"""
        import warnings
        # Suppress QGridLayoutEngine warnings
        warnings.filterwarnings('ignore', category=UserWarning, message='.*QGridLayoutEngine.*')
        
        layout = QGridLayout()

        layout.setContentsMargins(6, 5, 6, 5)

        # Testi leggibili mantenendo la GUI compatta
        self.setStyleSheet("""
            QLabel, QCheckBox {
                font-size: 10px;
            }
            QPushButton {
                font-size: 10px;
                min-height: 22px;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                font-size: 10px;
                min-height: 22px;
            }
        """)

        layout.setContentsMargins(6, 5, 6, 5)

        # Testi leggibili mantenendo la GUI compatta
        self.setStyleSheet("""
            QLabel, QCheckBox {
                font-size: 10px;
            }
            QPushButton {
                font-size: 10px;
                min-height: 22px;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                font-size: 10px;
                min-height: 22px;
            }
        """)

        layout.setContentsMargins(6, 5, 6, 5)

        # Testi leggibili mantenendo la GUI compatta
        self.setStyleSheet("""
            QLabel, QCheckBox {
                font-size: 10px;
            }
            QPushButton {
                font-size: 10px;
                min-height: 22px;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                font-size: 10px;
                min-height: 22px;
            }
        """)
        layout.setSpacing(3)
        # Configure column stretches for all used columns (0-5)
        layout.setColumnStretch(0, 0)  # Fixed width for labels
        layout.setColumnStretch(1, 0)  # Fixed width for controls
        layout.setColumnStretch(2, 0)  # Fixed width for buttons
        layout.setColumnStretch(3, 0)  # Fixed width for additional buttons
        layout.setColumnStretch(4, 0)  # Fixed width for additional buttons
        layout.setColumnStretch(5, 1)  # Stretch last column

        # Row 0: Frame skip
        layout.addWidget(QLabel("Frame skip:"), 0, 0)
        self.frame_skip_input = QSpinBox()
        self.frame_skip_input.setRange(1, 100)
        self.frame_skip_input.setValue(1)
        self.frame_skip_input.setSingleStep(1)
        self.frame_skip_input.setSuffix("x")
        self.frame_skip_input.setMaximumWidth(65)
        layout.addWidget(self.frame_skip_input, 0, 1)
        self.apply_frame_skip_btn = QPushButton("Apply")
        self.apply_frame_skip_btn.setMaximumWidth(50)
        layout.addWidget(self.apply_frame_skip_btn, 0, 2)

        # Row 1: Target FPS
        layout.addWidget(QLabel("Target FPS:"), 1, 0)
        self.vis_fps_input = QSpinBox()
        self.vis_fps_input.setRange(10, 120)
        self.vis_fps_input.setValue(60)
        self.vis_fps_input.setSingleStep(5)
        self.vis_fps_input.setSuffix(" Hz")
        self.vis_fps_input.setMaximumWidth(65)
        layout.addWidget(self.vis_fps_input, 1, 1)
        self.apply_vis_fps_btn = QPushButton("Apply")
        self.apply_vis_fps_btn.setMaximumWidth(50)
        layout.addWidget(self.apply_vis_fps_btn, 1, 2)

        # Row 1.5: Profiling overlay toggle
        self.show_profiling_checkbox = QCheckBox("Show Profiling Overlay")
        self.show_profiling_checkbox.setChecked(False)
        self.show_profiling_checkbox.setToolTip("Display timing information in overlay")
        layout.addWidget(self.show_profiling_checkbox, 2, 0, 1, 3)  # Span all columns

        # Row 3: Display toggles - horizontal layout within grid cell
        display_toggle_row = QHBoxLayout()
        self.show_velocity_checkbox = QCheckBox("Velocity")
        self.show_velocity_checkbox.setChecked(True)
        self.show_vorticity_checkbox = QCheckBox("Vorticity")
        self.show_vorticity_checkbox.setChecked(True)
        self.show_pressure_checkbox = QCheckBox("Pressure")
        self.show_pressure_checkbox.setChecked(False)
        self.show_density_checkbox = QCheckBox("Density")
        self.show_density_checkbox.setChecked(False)
        self.show_density_checkbox.setToolTip("Show density field (useful for 2-phase flow)")
        self.show_dye_checkbox = QCheckBox("Dye")
        self.show_dye_checkbox.setChecked(True)
        self.particle_mode_checkbox = QCheckBox("Particles")
        self.particle_mode_checkbox.setChecked(False)
        self.particle_mode_checkbox.setToolTip("Toggle between dye field and Lagrangian tracer particles")
        self.show_sdf_checkbox = QCheckBox("SDF Mask")
        self.show_sdf_checkbox.setChecked(False)
        self.show_streamlines_checkbox = QCheckBox("Streamlines")
        self.show_streamlines_checkbox.setChecked(False)
        self.show_quivers_checkbox = QCheckBox("Quivers")
        self.show_quivers_checkbox.setChecked(False)
        self.liquid_mode_checkbox = QCheckBox("Liquid Mode")
        self.liquid_mode_checkbox.setChecked(False)
        self.liquid_mode_checkbox.setToolTip("Enable liquid-like visualization effect with enhanced lighting")
        display_toggle_row = QVBoxLayout()
        display_toggle_row.setSpacing(2)

        display_row1 = QHBoxLayout()
        display_row1.setSpacing(3)
        display_row2 = QHBoxLayout()
        display_row2.setSpacing(3)

        display_row1.addWidget(self.show_velocity_checkbox)
        display_row1.addWidget(self.show_vorticity_checkbox)
        display_row1.addWidget(self.show_pressure_checkbox)
        display_row1.addWidget(self.show_density_checkbox)
        display_row1.addWidget(self.show_dye_checkbox)

        display_row2.addWidget(self.particle_mode_checkbox)
        display_row2.addWidget(self.show_sdf_checkbox)
        display_row2.addWidget(self.show_streamlines_checkbox)
        display_row2.addWidget(self.show_quivers_checkbox)
        display_row2.addWidget(self.liquid_mode_checkbox)

        display_row1.addStretch()
        display_row2.addStretch()

        display_toggle_row.addLayout(display_row1)
        display_toggle_row.addLayout(display_row2)

        layout.addLayout(display_toggle_row, 3, 0, 1, 3)

        # Row 4: Color scale options - compact 2 rows
        colorscale_row = QVBoxLayout()
        colorscale_row.setSpacing(2)

        colorscale_row1 = QHBoxLayout()
        colorscale_row1.setSpacing(3)

        self.log_colorscale_checkbox = QCheckBox("Log Color Scale")
        self.log_colorscale_checkbox.setChecked(True)

        self.spatial_colorscale_checkbox = QCheckBox("Spatial Weighting")
        self.spatial_colorscale_checkbox.setChecked(False)

        colorscale_row1.addWidget(self.log_colorscale_checkbox)
        colorscale_row1.addWidget(self.spatial_colorscale_checkbox)
        colorscale_row1.addStretch()

        colorscale_row2 = QHBoxLayout()
        colorscale_row2.setSpacing(3)

        self.adaptive_colorscale_checkbox = QCheckBox("Adaptive Scale")
        self.adaptive_colorscale_checkbox.setChecked(True)
        self.adaptive_colorscale_checkbox.setToolTip(
            "When enabled, color scales adjust automatically to data range. "
            "Disable to allow manual adjustment."
        )

        colorscale_row2.addWidget(self.adaptive_colorscale_checkbox)
        colorscale_row2.addStretch()

        colorscale_row.addLayout(colorscale_row1)
        colorscale_row.addLayout(colorscale_row2)

        layout.addLayout(colorscale_row, 4, 0, 1, 3)

        # Row 5: Visualization smoothing
        layout.addWidget(QLabel("Smooth:"), 5, 0)
        self.upscale_slider = QSlider(Qt.Orientation.Horizontal)
        self.upscale_slider.setRange(1, 10)
        self.upscale_slider.setValue(1)
        self.upscale_slider.setMaximumWidth(90)
        layout.addWidget(self.upscale_slider, 5, 1)
        self.upscale_label = QLabel("1x")
        layout.addWidget(self.upscale_label, 5, 2)

        # Row 5.5: Liquid height scale
        layout.addWidget(QLabel("Liquid Height:"), 6, 0)
        self.liquid_height_slider = QSlider(Qt.Orientation.Horizontal)
        self.liquid_height_slider.setRange(1, 50)
        self.liquid_height_slider.setValue(12)
        self.liquid_height_slider.setMaximumWidth(90)
        layout.addWidget(self.liquid_height_slider, 6, 1)
        self.liquid_height_label = QLabel("12")
        layout.addWidget(self.liquid_height_label, 6, 2)

        # Row 6.5: Liquid light direction X
        layout.addWidget(QLabel("Light X:"), 7, 0)
        self.liquid_light_x_slider = QSlider(Qt.Orientation.Horizontal)
        self.liquid_light_x_slider.setRange(-100, 100)
        self.liquid_light_x_slider.setValue(20)
        self.liquid_light_x_slider.setMaximumWidth(90)
        layout.addWidget(self.liquid_light_x_slider, 7, 1)
        self.liquid_light_x_label = QLabel("0.2")
        layout.addWidget(self.liquid_light_x_label, 7, 2)

        # Row 7.5: Liquid light direction Y
        layout.addWidget(QLabel("Light Y:"), 8, 0)
        self.liquid_light_y_slider = QSlider(Qt.Orientation.Horizontal)
        self.liquid_light_y_slider.setRange(-100, 100)
        self.liquid_light_y_slider.setValue(40)
        self.liquid_light_y_slider.setMaximumWidth(90)
        layout.addWidget(self.liquid_light_y_slider, 8, 1)
        self.liquid_light_y_label = QLabel("0.4")
        layout.addWidget(self.liquid_light_y_label, 8, 2)

        # Row 8.5: Liquid light direction Z
        layout.addWidget(QLabel("Light Z:"), 9, 0)
        self.liquid_light_z_slider = QSlider(Qt.Orientation.Horizontal)
        self.liquid_light_z_slider.setRange(-100, 100)
        self.liquid_light_z_slider.setValue(90)
        self.liquid_light_z_slider.setMaximumWidth(90)
        layout.addWidget(self.liquid_light_z_slider, 9, 1)
        self.liquid_light_z_label = QLabel("0.9")
        layout.addWidget(self.liquid_light_z_label, 9, 2)

        # Row 9.5: Liquid specular intensity
        layout.addWidget(QLabel("Specular:"), 10, 0)
        self.liquid_specular_slider = QSlider(Qt.Orientation.Horizontal)
        self.liquid_specular_slider.setRange(0, 100)
        self.liquid_specular_slider.setValue(50)
        self.liquid_specular_slider.setMaximumWidth(90)
        layout.addWidget(self.liquid_specular_slider, 10, 1)
        self.liquid_specular_label = QLabel("0.5")
        layout.addWidget(self.liquid_specular_label, 10, 2)

        # Row 10.5: Liquid diffuse intensity
        layout.addWidget(QLabel("Diffuse:"), 11, 0)
        self.liquid_diffuse_slider = QSlider(Qt.Orientation.Horizontal)
        self.liquid_diffuse_slider.setRange(0, 100)
        self.liquid_diffuse_slider.setValue(75)
        self.liquid_diffuse_slider.setMaximumWidth(120)
        layout.addWidget(self.liquid_diffuse_slider, 11, 1)
        self.liquid_diffuse_label = QLabel("0.75")
        layout.addWidget(self.liquid_diffuse_label, 11, 2)

        # Row 11.5: Liquid base color R
        layout.addWidget(QLabel("Color R:"), 12, 0)
        self.liquid_color_r_slider = QSlider(Qt.Orientation.Horizontal)
        self.liquid_color_r_slider.setRange(0, 100)
        self.liquid_color_r_slider.setValue(0)
        self.liquid_color_r_slider.setMaximumWidth(120)
        layout.addWidget(self.liquid_color_r_slider, 12, 1)
        self.liquid_color_r_label = QLabel("0.0")
        layout.addWidget(self.liquid_color_r_label, 12, 2)

        # Row 12.5: Liquid base color G
        layout.addWidget(QLabel("Color G:"), 13, 0)
        self.liquid_color_g_slider = QSlider(Qt.Orientation.Horizontal)
        self.liquid_color_g_slider.setRange(0, 100)
        self.liquid_color_g_slider.setValue(12)
        self.liquid_color_g_slider.setMaximumWidth(120)
        layout.addWidget(self.liquid_color_g_slider, 13, 1)
        self.liquid_color_g_label = QLabel("0.12")
        layout.addWidget(self.liquid_color_g_label, 13, 2)

        # Row 13.5: Liquid base color B
        layout.addWidget(QLabel("Color B:"), 14, 0)
        self.liquid_color_b_slider = QSlider(Qt.Orientation.Horizontal)
        self.liquid_color_b_slider.setRange(0, 100)
        self.liquid_color_b_slider.setValue(85)
        self.liquid_color_b_slider.setMaximumWidth(120)
        layout.addWidget(self.liquid_color_b_slider, 14, 1)
        self.liquid_color_b_label = QLabel("0.85")
        layout.addWidget(self.liquid_color_b_label, 14, 2)

        # Row 15: Velocity colormap
        layout.addWidget(QLabel("Velocity colormap:"), 15, 0)
        self.velocity_colormap_combo = QComboBox()
        self.velocity_colormap_combo.setMaximumWidth(150)
        self._populate_velocity_colormaps()
        layout.addWidget(self.velocity_colormap_combo, 15, 1, 1, 2)  # Span 2 columns

        # Row 16: Vorticity colormap
        layout.addWidget(QLabel("Vorticity colormap:"), 16, 0)
        self.vorticity_colormap_combo = QComboBox()
        self.vorticity_colormap_combo.setMaximumWidth(150)
        self._populate_vorticity_colormaps()
        layout.addWidget(self.vorticity_colormap_combo, 16, 1, 1, 2)  # Span 2 columns

        # Row 17: Pressure colormap
        layout.addWidget(QLabel("Pressure colormap:"), 17, 0)
        self.pressure_colormap_combo = QComboBox()
        self.pressure_colormap_combo.setMaximumWidth(150)
        self._populate_pressure_colormaps()
        layout.addWidget(self.pressure_colormap_combo, 17, 1, 1, 2)  # Span 2 columns

        # Row 18-19: Export / State buttons - compact 2-row layout
        export_row1 = QHBoxLayout()
        export_row1.setSpacing(4)
        export_row1.setSpacing(4)
        self.export_btn = QPushButton("Export Frame")
        self.export_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.export_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.export_btn.setMaximumWidth(95)
        export_row1.addWidget(self.export_btn)

        self.record_btn = QPushButton("Record")
        self.record_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.record_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.record_btn.setMaximumWidth(70)
        export_row1.addWidget(self.record_btn)

        self.save_video_btn = QPushButton("Save Video")
        self.save_video_btn.setEnabled(False)
        self.save_video_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.save_video_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.save_video_btn.setMaximumWidth(85)
        export_row1.addWidget(self.save_video_btn)
        export_row1.addStretch()

        layout.addLayout(export_row1, 18, 0, 1, 3)

        export_row2 = QHBoxLayout()
        export_row2.setSpacing(4)
        export_row2.setSpacing(4)
        self.save_state_btn = QPushButton("Save State")
        self.save_state_btn.setEnabled(True)
        self.save_state_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.save_state_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.save_state_btn.setMaximumWidth(85)
        export_row2.addWidget(self.save_state_btn)

        self.load_state_btn = QPushButton("Load State")
        self.load_state_btn.setEnabled(True)
        self.load_state_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.load_state_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.load_state_btn.setMaximumWidth(85)
        export_row2.addWidget(self.load_state_btn)
        export_row2.addStretch()

        layout.addLayout(export_row2, 19, 0, 1, 3)

        # Row 20-21: Auto-scale buttons - compact 2-row layout
        autoscale_row1 = QHBoxLayout()
        autoscale_row1.setSpacing(4)
        autoscale_row1.setSpacing(4)
        autoscale_row1.addWidget(QLabel("Auto-scale:"))

        self.autofit_velocity_btn = QPushButton("Velocity")
        self.autofit_velocity_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.autofit_velocity_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.autofit_velocity_btn.setMaximumWidth(75)
        autoscale_row1.addWidget(self.autofit_velocity_btn)

        self.autofit_vorticity_btn = QPushButton("Vorticity")
        self.autofit_vorticity_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.autofit_vorticity_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.autofit_vorticity_btn.setMaximumWidth(75)
        autoscale_row1.addWidget(self.autofit_vorticity_btn)
        autoscale_row1.addStretch()

        layout.addLayout(autoscale_row1, 20, 0, 1, 3)

        autoscale_row2 = QHBoxLayout()
        autoscale_row2.setSpacing(4)
        autoscale_row2.setSpacing(4)
        self.autofit_pressure_btn = QPushButton("Pressure")
        self.autofit_pressure_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.autofit_pressure_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.autofit_pressure_btn.setMaximumWidth(75)
        autoscale_row2.addWidget(self.autofit_pressure_btn)

        self.autofit_dye_btn = QPushButton("Dye")
        self.autofit_dye_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.autofit_dye_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.autofit_dye_btn.setMaximumWidth(50)
        autoscale_row2.addWidget(self.autofit_dye_btn)

        self.autofit_all_btn = QPushButton("All")
        self.autofit_all_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.autofit_all_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.autofit_all_btn.setMaximumWidth(50)
        autoscale_row2.addWidget(self.autofit_all_btn)
        autoscale_row2.addStretch()

        layout.addLayout(autoscale_row2, 21, 0, 1, 3)

        self.setLayout(layout)

    def _populate_velocity_colormaps(self):
        """Populate velocity colormap dropdown"""
        velocity_colormaps = [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo',
            'CET-C1', 'CET-C2', 'CET-C3', 'CET-C4', 'CET-C5', 'CET-C6', 'CET-C7',
            'CET-D1', 'CET-D2', 'CET-D3', 'CET-D4', 'CET-D6', 'CET-D7', 'CET-D8',
            'CET-D9', 'CET-D10', 'CET-D11', 'CET-D12', 'CET-D13',
            'CET-L1', 'CET-L2', 'CET-L3', 'CET-L4', 'CET-L5', 'CET-L6', 'CET-L7',
            'CET-L8', 'CET-L9', 'CET-L10', 'CET-L11', 'CET-L12', 'CET-L13', 'CET-L14',
            'CET-L15', 'CET-L16', 'CET-L17', 'CET-L18', 'CET-L19',
            'PAL-relaxed', 'PAL-relaxed_bright'
        ]
        self.velocity_colormap_combo.addItems(velocity_colormaps)
        config = ConfigManager()
        self.velocity_colormap_combo.setCurrentText(config.viz_config.default_velocity_colormap)

    def _populate_vorticity_colormaps(self):
        """Populate vorticity colormap dropdown"""
        vorticity_colormaps = [
            'CET-CBC1', 'coolwarm', 'RdBu', 'seismic', 'bwr', 'PiYG', 'PRGn', 'BrBG',
            'CET-CBC2', 'CET-CBD1', 'CET-CBL1', 'CET-CBL2',
            'CET-CBTC1', 'CET-CBTC2', 'CET-CBTD1', 'CET-CBTL1', 'CET-CBTL2',
            'CET-I1', 'CET-I2', 'CET-I3',
            'CET-R1', 'CET-R2', 'CET-R3', 'CET-R4',
            '--- Sequential (Magnitude) ---',
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'PAL-relaxed', 'PAL-relaxed_bright'
        ]
        self.vorticity_colormap_combo.addItems(vorticity_colormaps)
        config = ConfigManager()
        self.vorticity_colormap_combo.setCurrentText(config.viz_config.default_vorticity_colormap)

    def _populate_pressure_colormaps(self):
        """Populate pressure colormap dropdown"""
        pressure_colormaps = [
            'CET-CBC1', 'coolwarm', 'RdBu', 'seismic', 'bwr', 'PiYG', 'PRGn', 'BrBG',
            'CET-CBC2', 'CET-CBD1', 'CET-CBL1', 'CET-CBL2',
            'CET-CBTC1', 'CET-CBTC2', 'CET-CBTD1', 'CET-CBTL1', 'CET-CBTL2',
            'CET-I1', 'CET-I2', 'CET-I3',
            'CET-R1', 'CET-R2', 'CET-R3', 'CET-R4',
            '--- Sequential (Magnitude) ---',
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'PAL-relaxed', 'PAL-relaxed_bright'
        ]
        self.pressure_colormap_combo.addItems(pressure_colormaps)
        config = ConfigManager()
        # Default to RdBu for pressure (diverging colormap suitable for pressure)
        self.pressure_colormap_combo.setCurrentText('RdBu')
