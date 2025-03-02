import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors
from scipy.integrate import solve_ivp
import matplotlib as mpl
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from matplotlib import cm
import matplotlib.gridspec as gridspec

class AdvancedRailGunSimulation:
    """
    Advanced 3D simulation of a rail gun with realistic physics and visualization.
    """
    def __init__(self, 
                 # Rail gun dimensions
                 rail_length=5.0,        # meters
                 rail_width=0.05,        # meters
                 rail_height=0.02,       # meters
                 rail_separation=0.15,   # meters
                 # Power and electrical parameters
                 capacitor_bank_energy=500000,  # Joules (500 kJ)
                 capacitor_voltage=5000,        # Volts
                 rail_resistivity=1.68e-8,      # Ohm*m (copper)
                 contact_resistance=1e-4,       # Ohms
                 inductance_gradient=0.5e-6,    # H/m
                 # Projectile parameters
                 projectile_mass=0.1,           # kg
                 projectile_density=7800,       # kg/m³ (steel)
                 projectile_resistivity=9.71e-8, # Ohm*m (iron)
                 # Simulation parameters
                 initial_position=0.01,         # m
                 initial_velocity=0.0,          # m/s
                 simulation_time=0.1,           # seconds
                 time_steps=1000,               # number of time steps
                 # Physical constants
                 mu_0=4*np.pi*1e-7,            # H/m (magnetic permeability of vacuum)
                 g=9.81,                        # m/s² (gravitational acceleration)
                 air_density=1.225,             # kg/m³
                 drag_coefficient=0.1,          # dimensionless
                 # Material properties for thermal simulation
                 specific_heat_capacity=450,    # J/(kg*K) for steel
                 thermal_conductivity=50,       # W/(m*K) for steel
                 ambient_temperature=293        # K (20°C)
                ):
        
        # Store all parameters
        self.rail_length = rail_length
        self.rail_width = rail_width
        self.rail_height = rail_height
        self.rail_separation = rail_separation
        
        self.capacitor_bank_energy = capacitor_bank_energy
        self.capacitor_voltage = capacitor_voltage
        self.rail_resistivity = rail_resistivity
        self.contact_resistance = contact_resistance
        self.inductance_gradient = inductance_gradient
        
        self.projectile_mass = projectile_mass
        self.projectile_density = projectile_density
        self.projectile_resistivity = projectile_resistivity
        
        self.initial_position = initial_position
        self.initial_velocity = initial_velocity
        self.simulation_time = simulation_time
        self.time_steps = time_steps
        
        self.mu_0 = mu_0
        self.g = g
        self.air_density = air_density
        self.drag_coefficient = drag_coefficient
        
        self.specific_heat_capacity = specific_heat_capacity
        self.thermal_conductivity = thermal_conductivity
        self.ambient_temperature = ambient_temperature
        
        # Derived parameters
        self.capacitance = 2 * self.capacitor_bank_energy / (self.capacitor_voltage ** 2)
        self.rail_cross_section = rail_width * rail_height
        self.rail_resistance_per_meter = rail_resistivity / self.rail_cross_section
        
        # Calculate projectile dimensions based on mass and density
        self.projectile_volume = self.projectile_mass / self.projectile_density
        # Assume projectile is rectangular with width = rail_separation + margin
        self.projectile_width = self.rail_separation * 1.2
        self.projectile_height = self.rail_height * 2
        self.projectile_length = self.projectile_volume / (self.projectile_width * self.projectile_height)
        
        # Cross-sectional area for drag calculation
        self.projectile_area = self.projectile_width * self.projectile_height
        
        # Initialize simulation results containers
        self.time_points = None
        self.positions = None
        self.velocities = None
        self.accelerations = None
        self.currents = None
        self.forces = None
        self.energies = None
        self.temperatures = None
        self.voltages = None
        self.capacitor_charge = None
        
        # Simulation flags
        self.simulation_run = False
        self.include_air_resistance = True
        self.include_thermal_effects = True
        self.include_capacitor_discharge = True
        
    def compute_total_resistance(self, position):
        """Calculate the total circuit resistance based on position."""
        # Rail resistance (based on length of the current path)
        rail_resistance = 2 * position * self.rail_resistance_per_meter
        
        # Return total resistance including contact resistance
        return rail_resistance + self.contact_resistance
    
    def compute_inductance(self, position):
        """Calculate the circuit inductance based on position."""
        # L = L' * x, where L' is the inductance gradient
        return self.inductance_gradient * position
    
    def compute_current(self, time, position):
        """Calculate the current based on capacitor discharge and circuit parameters."""
        if not self.include_capacitor_discharge:
            # Constant current approximation
            return np.sqrt(self.capacitor_bank_energy / self.compute_inductance(position))
        
        # Find the time index
        if time <= 0:
            return 0
        
        time_index = min(int(time / self.simulation_time * self.time_steps), len(self.currents) - 1)
        return self.currents[time_index]
    
    def compute_lorentz_force(self, current, velocity):
        """Calculate the Lorentz force on the projectile."""
        # F = 0.5 * L' * I²
        return 0.5 * self.inductance_gradient * current ** 2
    
    def compute_drag_force(self, velocity):
        """Calculate the aerodynamic drag force."""
        if not self.include_air_resistance or velocity <= 0:
            return 0
        
        # F_drag = 0.5 * ρ * v² * Cd * A
        return 0.5 * self.air_density * velocity ** 2 * self.drag_coefficient * self.projectile_area
    
    def compute_ohmic_heating(self, current, position):
        """Calculate the heat generated by Joule heating."""
        if not self.include_thermal_effects:
            return 0
        
        # Power = I² * R, focusing on projectile heating
        contact_power = current ** 2 * self.contact_resistance
        
        # Convert power to temperature increase (Q = m * c * ΔT)
        # Assuming all heat goes to the projectile over the time step
        return contact_power / (self.projectile_mass * self.specific_heat_capacity)
    
    def system_dynamics(self, t, state):
        """
        Define the system dynamics for the rail gun.
        
        state components:
        [0]: position (x)
        [1]: velocity (v)
        [2]: capacitor charge (q) - if capacitor discharge is included
        [3]: temperature (T) - if thermal effects are included
        """
        # Unpack state variables
        position = state[0]
        velocity = state[1]
        
        # Default values
        dq_dt = 0
        dT_dt = 0
        
        # If projectile has left the rails, only gravity and drag apply
        if position > self.rail_length:
            current = 0
            lorentz_force = 0
        else:
            if self.include_capacitor_discharge:
                charge = state[2]
                # Calculate circuit parameters
                resistance = self.compute_total_resistance(position)
                inductance = self.compute_inductance(position)
                
                # Calculate current using I = q/C
                current = charge / self.capacitance if self.capacitance > 0 else 0
                
                # Charge dynamics: dq/dt = -I
                dq_dt = -current
                
                # Calculate back-EMF from projectile motion
                back_emf = velocity * self.inductance_gradient * current
                
                # Circuit equation: L*dI/dt + R*I + back_EMF = V
                di_dt = (self.capacitor_voltage * charge / self.capacitance - resistance * current - back_emf) / inductance
                
                # Update current calculation with di_dt effect for next step
                current_derivative_effect = di_dt * (self.simulation_time / self.time_steps)
                current = max(0, current + current_derivative_effect)
            else:
                # Use simplified constant current approximation
                current = self.compute_current(t, position)
            
            # Calculate Lorentz force
            lorentz_force = self.compute_lorentz_force(current, velocity)
        
        # Calculate drag force (always applies)
        drag_force = self.compute_drag_force(velocity)
        
        # Total force
        total_force = lorentz_force - drag_force
        
        # Acceleration
        acceleration = total_force / self.projectile_mass
        
        # Temperature dynamics
        if self.include_thermal_effects and position <= self.rail_length:
            heating_rate = self.compute_ohmic_heating(current, position)
            cooling_rate = 0.01 * (state[3] - self.ambient_temperature)  # Simple cooling model
            dT_dt = heating_rate - cooling_rate
        
        # Return state derivatives [dx/dt, dv/dt, dq/dt, dT/dt]
        derivatives = [velocity, acceleration]
        
        if self.include_capacitor_discharge:
            derivatives.append(dq_dt)
            
        if self.include_thermal_effects:
            derivatives.append(dT_dt)
            
        return derivatives
    
    def run_simulation(self):
        """Run the complete rail gun simulation."""
        # Initialize state vector
        initial_state = [self.initial_position, self.initial_velocity]
        
        # Add capacitor charge if discharge model is included
        if self.include_capacitor_discharge:
            initial_charge = self.capacitance * self.capacitor_voltage
            initial_state.append(initial_charge)
        
        # Add temperature if thermal effects are included
        if self.include_thermal_effects:
            initial_state.append(self.ambient_temperature)
        
        # Time span for the simulation
        t_span = (0, self.simulation_time)
        t_eval = np.linspace(0, self.simulation_time, self.time_steps)
        
        # Solve the system of differential equations
        result = solve_ivp(
            self.system_dynamics,
            t_span,
            initial_state,
            method='RK45',
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-9
        )
        
        # Extract results
        self.time_points = result.t
        self.positions = result.y[0]
        self.velocities = result.y[1]
        
        # Initialize other result arrays
        self.accelerations = np.zeros_like(self.velocities)
        self.currents = np.zeros_like(self.velocities)
        self.forces = np.zeros_like(self.velocities)
        self.lorentz_forces = np.zeros_like(self.velocities)
        self.drag_forces = np.zeros_like(self.velocities)
        self.energies = np.zeros_like(self.velocities)
        self.voltages = np.zeros_like(self.velocities)
        
        # Extract capacitor charge if included
        if self.include_capacitor_discharge:
            self.capacitor_charge = result.y[2]
            charge_index = 2
        else:
            self.capacitor_charge = self.capacitance * self.capacitor_voltage * np.ones_like(self.velocities)
            charge_index = -1
        
        # Extract temperature if included
        if self.include_thermal_effects:
            self.temperatures = result.y[charge_index + 1] if charge_index > 0 else result.y[2]
        else:
            self.temperatures = self.ambient_temperature * np.ones_like(self.velocities)
        
        # Calculate derived quantities
        for i in range(len(self.time_points)):
            position = self.positions[i]
            velocity = self.velocities[i]
            
            # Calculate current
            if self.include_capacitor_discharge:
                self.currents[i] = self.capacitor_charge[i] / self.capacitance if self.capacitance > 0 else 0
            else:
                self.currents[i] = self.compute_current(self.time_points[i], position)
            
            # Calculate forces
            if position <= self.rail_length:
                self.lorentz_forces[i] = self.compute_lorentz_force(self.currents[i], velocity)
            else:
                self.lorentz_forces[i] = 0
                
            self.drag_forces[i] = self.compute_drag_force(velocity)
            self.forces[i] = self.lorentz_forces[i] - self.drag_forces[i]
            
            # Calculate acceleration
            self.accelerations[i] = self.forces[i] / self.projectile_mass
            
            # Calculate kinetic energy
            self.energies[i] = 0.5 * self.projectile_mass * velocity ** 2
            
            # Calculate voltage
            resistance = self.compute_total_resistance(position)
            inductance = self.compute_inductance(position)
            back_emf = velocity * self.inductance_gradient * self.currents[i] if position <= self.rail_length else 0
            self.voltages[i] = self.currents[i] * resistance + back_emf
        
        # Mark simulation as run
        self.simulation_run = True
        
        # Calculate exit conditions if projectile leaves the rails
        exit_indices = np.where(self.positions >= self.rail_length)[0]
        if len(exit_indices) > 0:
            exit_index = exit_indices[0]
            exit_time = self.time_points[exit_index]
            exit_velocity = self.velocities[exit_index]
            exit_energy = self.energies[exit_index]
            exit_temperature = self.temperatures[exit_index]
        else:
            exit_time = None
            exit_velocity = None
            exit_energy = None
            exit_temperature = None
        
        # Calculate efficiency
        initial_energy = self.capacitor_bank_energy
        final_energy = self.energies[-1]
        efficiency = (final_energy / initial_energy) * 100 if initial_energy > 0 else 0
        
        # Return simulation results summary
        results = {
            'time': self.time_points,
            'position': self.positions,
            'velocity': self.velocities,
            'acceleration': self.accelerations,
            'current': self.currents,
            'force': self.forces,
            'energy': self.energies,
            'temperature': self.temperatures,
            'exit_time': exit_time,
            'exit_velocity': exit_velocity,
            'exit_energy': exit_energy,
            'exit_temperature': exit_temperature,
            'efficiency': efficiency
        }
        
        return results
    
    # === 3D Model Creation Methods ===
    
    def create_rail_vertices(self, is_upper_rail=True):
        """Create vertices for a 3D rail."""
        # Rail center y-position
        rail_y = self.rail_separation / 2 if is_upper_rail else -self.rail_separation / 2
        
        # Rail dimensions
        rail_width = self.rail_width
        rail_height = self.rail_height
        
        # Create vertices for rectangular prism
        vertices = np.array([
            # Front face
            [0, rail_y - rail_width/2, 0],
            [0, rail_y + rail_width/2, 0],
            [0, rail_y + rail_width/2, rail_height],
            [0, rail_y - rail_width/2, rail_height],
            
            # Back face
            [self.rail_length, rail_y - rail_width/2, 0],
            [self.rail_length, rail_y + rail_width/2, 0],
            [self.rail_length, rail_y + rail_width/2, rail_height],
            [self.rail_length, rail_y - rail_width/2, rail_height]
        ])
        
        # Define faces
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Front
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # Back
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Bottom
            [vertices[3], vertices[2], vertices[6], vertices[7]]   # Top
        ]
        
        return faces
    
    def create_projectile_vertices(self, position):
        """Create vertices for a 3D projectile at given position."""
        # Dimensions
        width = self.projectile_width
        height = self.projectile_height
        length = self.projectile_length
        
        # Calculate front face position
        front_pos = position - length / 2
        rear_pos = position + length / 2
        
        # Create vertices for rectangular prism
        vertices = np.array([
            # Front face
            [front_pos, -width/2, 0],
            [front_pos, width/2, 0],
            [front_pos, width/2, height],
            [front_pos, -width/2, height],
            
            # Back face
            [rear_pos, -width/2, 0],
            [rear_pos, width/2, 0],
            [rear_pos, width/2, height],
            [rear_pos, -width/2, height]
        ])
        
        # Define faces
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Front
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # Back
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Bottom
            [vertices[3], vertices[2], vertices[6], vertices[7]]   # Top
        ]
        
        return faces
    
    def create_capacitor_bank_vertices(self):
        """Create vertices for capacitor bank."""
        # Dimensions for capacitor bank
        cap_width = self.rail_separation * 2
        cap_height = self.rail_height * 4
        cap_depth = cap_width / 2
        
        # Position (behind the rails)
        cap_x = -cap_depth * 1.5
        cap_y = 0
        cap_z = 0
        
        # Create vertices
        vertices = np.array([
            # Front face
            [cap_x, cap_y - cap_width/2, cap_z],
            [cap_x, cap_y + cap_width/2, cap_z],
            [cap_x, cap_y + cap_width/2, cap_z + cap_height],
            [cap_x, cap_y - cap_width/2, cap_z + cap_height],
            
            # Back face
            [cap_x - cap_depth, cap_y - cap_width/2, cap_z],
            [cap_x - cap_depth, cap_y + cap_width/2, cap_z],
            [cap_x - cap_depth, cap_y + cap_width/2, cap_z + cap_height],
            [cap_x - cap_depth, cap_y - cap_width/2, cap_z + cap_height]
        ])
        
        # Define faces
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Front
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # Back
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Bottom
            [vertices[3], vertices[2], vertices[6], vertices[7]]   # Top
        ]
        
        return faces

    def create_magnetic_field_lines(self, position, current):
        """Create magnetic field lines around the current position."""
        field_lines = []
        
        # Only show field if projectile is on the rails and current is flowing
        if position > self.rail_length or current <= 0:
            return field_lines
        
        # Normalize current for visual scaling
        current_norm = min(1.0, current / 100000)
        
        # Number of field lines based on current
        num_lines = max(4, int(8 * current_norm))
        
        # Create circular field lines around each rail
        radius_range = np.linspace(self.rail_separation * 0.5, 
                                  self.rail_separation * (1.0 + current_norm),
                                  num_lines//2)
        
        for radius in radius_range:
            theta = np.linspace(0, 2 * np.pi, 50)
            
            # Upper rail field
            rail_y = self.rail_separation / 2
            x_upper = np.ones_like(theta) * position
            y_upper = rail_y + radius * np.cos(theta)
            z_upper = self.rail_height / 2 + radius * np.sin(theta)
            
            # Lower rail field
            rail_y = -self.rail_separation / 2
            x_lower = np.ones_like(theta) * position
            y_lower = rail_y + radius * np.cos(theta)
            z_lower = self.rail_height / 2 + radius * np.sin(theta)
            
            field_lines.append((x_upper, y_upper, z_upper))
            field_lines.append((x_lower, y_lower, z_lower))
        
        return field_lines
    
    def create_plasma_trail(self, position, temperature, velocity):
        """Create plasma trail behind the projectile."""
        if position <= self.initial_position or temperature <= self.ambient_temperature + 100:
            return [], []
        
        # Calculate trail length based on velocity and temperature
        vel_factor = min(1.0, velocity / 1000)
        temp_factor = min(1.0, (temperature - self.ambient_temperature) / 1000)
        trail_length = min(position - self.initial_position, 
                          self.rail_length * 0.3 * vel_factor * temp_factor)
        
        # No trail if factors are too small
        if trail_length < 0.05:
            return [], []
        
        # Create trail points
        num_points = 20
        x_trail = np.linspace(position - trail_length, position, num_points)
        
        # Add some randomness for plasma effect
        y_spread = self.rail_separation * 0.3 * temp_factor
        z_spread = self.rail_height * temp_factor
        
        y_trail = np.random.normal(0, y_spread, num_points)
        z_trail = np.random.normal(self.rail_height, z_spread, num_points)
        
        # Color gradient based on temperature
        colors = np.zeros((num_points, 4))  # RGBA
        for i in range(num_points):
            # Position-based color (closer to projectile = hotter)
            pos_factor = i / num_points
            
            # Start with red-orange color for plasma
            colors[i, 0] = 1.0  # Red
            colors[i, 1] = 0.6 * pos_factor  # Green component
            colors[i, 2] = 0.1 * pos_factor  # Blue component
            
            # Alpha (transparency) decreases with distance from projectile
            colors[i, 3] = (1 - pos_factor) ** 0.7
        
        return (x_trail, y_trail, z_trail), colors
    
    # === Visualization Methods ===
    
    def create_3d_plot(self):
        """Create the base 3D plot for rail gun visualization."""
        # Create figure and 3D axes with appropriate size
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set aspect ratio to make visualization clearer
        ax.set_box_aspect([2, 1, 1])
        
        # Set plot limits with some padding
        ax.set_xlim(-self.rail_length * 0.2, self.rail_length * 1.2)
        ax.set_ylim(-self.rail_separation * 3, self.rail_separation * 3)
        ax.set_zlim(-self.rail_height * 2, self.rail_height * 6)
        
        # Set labels
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Advanced 3D Rail Gun Simulation')
        
        return fig, ax
    
    def add_static_components(self, ax):
        """Add static components (rails, etc.) to the 3D plot."""
        # Create and add rails
        upper_rail_faces = self.create_rail_vertices(is_upper_rail=True)
        lower_rail_faces = self.create_rail_vertices(is_upper_rail=False)
        
        upper_rail = Poly3DCollection(upper_rail_faces, alpha=0.9, linewidth=1, edgecolor='k')
        upper_rail.set_facecolor('silver')
        lower_rail = Poly3DCollection(lower_rail_faces, alpha=0.9, linewidth=1, edgecolor='k')
        lower_rail.set_facecolor('silver')
        
        ax.add_collection3d(upper_rail)
        ax.add_collection3d(lower_rail)
        
        # Create and add capacitor bank
        capacitor_faces = self.create_capacitor_bank_vertices()
        capacitor = Poly3DCollection(capacitor_faces, alpha=0.8, linewidth=1, edgecolor='k')
        capacitor.set_facecolor('darkblue')
        ax.add_collection3d(capacitor)
        
        # Add base plate
        plate_width = self.rail_separation * 3
        base_x = [0, 0, self.rail_length, self.rail_length]
        base_y = [-plate_width/2, plate_width/2, plate_width/2, -plate_width/2]
        base_z = [-0.01, -0.01, -0.01, -0.01]
        base_vertices = [list(zip(base_x, base_y, base_z))]
        base = Poly3DCollection(base_vertices, alpha=0.5, linewidth=1, edgecolor='k')
        base.set_facecolor('gray')
        ax.add_collection3d(base)
        
        # Add connection wires from capacitor to rails
        cap_x = self.create_capacitor_bank_vertices()[0][0][0]  # Front center of capacitor
        cap_y = 0
        cap_z = self.rail_height * 2
        
        # Upper rail connection
        upper_rail_y = self.rail_separation / 2
        ax.plot([cap_x, 0], [cap_y, upper_rail_y], [cap_z, self.rail_height/2], 'k-', linewidth=2)
        
        # Lower rail connection
        lower_rail_y = -self.rail_separation / 2
        ax.plot([cap_x, 0], [cap_y, lower_rail_y], [cap_z, self.rail_height/2], 'k-', linewidth=2)
        
        # Add distance markers along the rails
        for x in np.linspace(0, self.rail_length, 6):
            # Skip the end positions
            if x == 0 or x == self.rail_length:
                continue
                
            # Add small marker lines
            marker_len = self.rail_separation * 0.2
            ax.plot([x, x], [-marker_len, marker_len], [0, 0], 'k-', linewidth=1, alpha=0.7)
            
            # Add text label
            ax.text(x, 0, -0.05, f'{x:.1f}m', fontsize=8, ha='center')
    
    def visualize_static_3d(self, time_index=None):
        """Create a static 3D visualization of the rail gun."""
        if not self.simulation_run:
            self.run_simulation()
            
        # Choose time index if not specified
        if time_index is None:
            # Find the index where the projectile is approximately in the middle of the rails
            mid_position = self.rail_length / 2
            distances = np.abs(self.positions - mid_position)
            time_index = np.argmin(distances)
        
        # Get values at the specified time
        position = self.positions[time_index]
        velocity = self.velocities[time_index]
        current = self.currents[time_index]
        temperature = self.temperatures[time_index]
        time = self.time_points[time_index]
        
        # Create 3D figure and axes
        fig, ax = self.create_3d_plot()
        
        # Add static components
        self.add_static_components(ax)
        
        # Add projectile
        projectile_faces = self.create_projectile_vertices(position)
        
        # Color projectile based on temperature
        temp_norm = min(1.0, max(0, (temperature - self.ambient_temperature) / 1000))
        projectile_color = mcolors.to_rgba(plt.cm.plasma(temp_norm))
        
        projectile = Poly3DCollection(projectile_faces, alpha=0.9, linewidth=1, edgecolor='k')
        projectile.set_facecolor(projectile_color)
        ax.add_collection3d(projectile)
        
        # Add magnetic field lines if projectile is on the rails
        if position <= self.rail_length:
            field_lines = self.create_magnetic_field_lines(position, current)
            for x, y, z in field_lines:
                ax.plot(x, y, z, 'b-', alpha=0.5, linewidth=1)
        
        # Add plasma trail if temperature is high enough
        plasma_points, plasma_colors = self.create_plasma_trail(position, temperature, velocity)
        if len(plasma_points) > 0:
            x_trail, y_trail, z_trail = plasma_points
            ax.scatter(x_trail, y_trail, z_trail, c=plasma_colors, s=50, marker='o')
        
        # Add trajectory line
        if time_index > 0:
            ax.plot(self.positions[:time_index+1], 
                    np.zeros_like(self.positions[:time_index+1]), 
                    self.rail_height * 1.5 * np.ones_like(self.positions[:time_index+1]), 
                    'r--', linewidth=1.5, label='Trajectory')
        
        # Add data annotation
        ax.text2D(0.02, 0.97, f'Time: {time:.3f} s', transform=ax.transAxes, fontsize=9)
        ax.text2D(0.02, 0.94, f'Position: {position:.2f} m', transform=ax.transAxes, fontsize=9)
        ax.text2D(0.02, 0.91, f'Velocity: {velocity:.1f} m/s', transform=ax.transAxes, fontsize=9)
        ax.text2D(0.02, 0.88, f'Current: {current:.1f} A', transform=ax.transAxes, fontsize=9)
        ax.text2D(0.02, 0.85, f'Temperature: {temperature-273.15:.1f} °C', transform=ax.transAxes, fontsize=9)
        
        # Add legend
        ax.legend()
        
        return fig, ax
    
    def animate_3d_simulation(self, interval=50, save=False, filename='advanced_railgun.gif'):
        """Create a 3D animation of the rail gun simulation."""
        if not self.simulation_run:
            self.run_simulation()
        
        # Create figure and 3D axes
        fig, ax = self.create_3d_plot()
        
        # Add static components
        self.add_static_components(ax)
        
        # Initialize projectile (will be updated in animation)
        projectile_faces = self.create_projectile_vertices(self.initial_position)
        projectile = Poly3DCollection(projectile_faces, alpha=0.9, linewidth=1, edgecolor='k')
        projectile.set_facecolor('darkred')
        ax.add_collection3d(projectile)
        
        # Initialize magnetic field lines
        field_lines_data = self.create_magnetic_field_lines(self.initial_position, self.currents[0])
        field_lines_objects = []
        
        for x, y, z in field_lines_data:
            line, = ax.plot(x, y, z, 'b-', alpha=0.5, linewidth=1)
            field_lines_objects.append(line)
        
        # Initialize plasma trail
        plasma_scatter = ax.scatter([], [], [], c='red', s=30, alpha=0.7)
        
        # Initialize trajectory line
        trajectory, = ax.plot([], [], [], 'r--', linewidth=1.5, label='Trajectory')
        
        # Text annotations for simulation data
        time_text = ax.text2D(0.02, 0.97, '', transform=ax.transAxes, fontsize=9)
        position_text = ax.text2D(0.02, 0.94, '', transform=ax.transAxes, fontsize=9)
        velocity_text = ax.text2D(0.02, 0.91, '', transform=ax.transAxes, fontsize=9)
        current_text = ax.text2D(0.02, 0.88, '', transform=ax.transAxes, fontsize=9)
        temperature_text = ax.text2D(0.02, 0.85, '', transform=ax.transAxes, fontsize=9)
        
        # Add colorbar for temperature
        cmap = plt.cm.plasma
        norm = mcolors.Normalize(vmin=self.ambient_temperature, vmax=self.ambient_temperature + 1000)
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, 
                           label='Temperature (K)', pad=0.1, shrink=0.7)
        
        # Interpolate for smoother animation
        num_frames = min(100, len(self.time_points))
        t_indices = np.linspace(0, len(self.time_points)-1, num_frames).astype(int)
        
        # Function to update the plot for each animation frame
        def update(frame):
            i = t_indices[frame]
            
            # Get current values
            pos = self.positions[i]
            vel = self.velocities[i]
            curr = self.currents[i]
            temp = self.temperatures[i]
            t = self.time_points[i]
            
            # Update projectile position
            new_projectile_faces = self.create_projectile_vertices(pos)
            projectile.set_verts(new_projectile_faces)
            
            # Update projectile color based on temperature
            temp_norm = min(1.0, max(0, (temp - self.ambient_temperature) / 1000))
            projectile.set_facecolor(plt.cm.plasma(temp_norm))
            
            # Update magnetic field lines
            new_field_lines = self.create_magnetic_field_lines(pos, curr)
            
            # Clear old field lines if needed
            if len(new_field_lines) == 0:
                for line in field_lines_objects:
                    line.set_data([], [])
                    line.set_3d_properties([])
            else:
                # Update existing field lines
                for j, (x, y, z) in enumerate(new_field_lines):
                    if j < len(field_lines_objects):
                        field_lines_objects[j].set_data(x, y)
                        field_lines_objects[j].set_3d_properties(z)
            
            # Update plasma trail
            plasma_points, plasma_colors = self.create_plasma_trail(pos, temp, vel)
            if len(plasma_points) > 0:
                x_trail, y_trail, z_trail = plasma_points
                plasma_scatter._offsets3d = (x_trail, y_trail, z_trail)
                plasma_scatter.set_color(plasma_colors)
                plasma_scatter.set_alpha(plasma_colors[:, 3])
                plasma_scatter.set_sizes(np.ones_like(x_trail) * 30)
            else:
                plasma_scatter._offsets3d = ([], [], [])
            
            # Update trajectory
            trajectory.set_data(self.positions[:i+1], np.zeros_like(self.positions[:i+1]))
            trajectory.set_3d_properties(self.rail_height * 1.5 * np.ones_like(self.positions[:i+1]))
            
            # Update text
            time_text.set_text(f'Time: {t:.3f} s')
            position_text.set_text(f'Position: {pos:.2f} m')
            velocity_text.set_text(f'Velocity: {vel:.1f} m/s')
            current_text.set_text(f'Current: {curr:.1f} A')
            temperature_text.set_text(f'Temperature: {temp-273.15:.1f} °C')
            
            # Adjust camera to follow projectile if it gets far
            if pos > self.rail_length * 0.8:
                ax.set_xlim(pos - self.rail_length * 0.5, pos + self.rail_length * 0.5)
                
            # Return all objects that were updated
            return [projectile, plasma_scatter, trajectory, time_text, position_text, 
                    velocity_text, current_text, temperature_text] + field_lines_objects
        
        # Create animation
        ani = FuncAnimation(
            fig, update, frames=num_frames,
            interval=interval, blit=False
        )
        
        if save:
            print(f"Saving animation to {filename}...")
            ani.save(filename, writer='pillow', fps=30)
            print("Animation saved!")
        
        return ani

    def create_dashboard(self):
        """Create an interactive dashboard with simulation controls and results visualization."""
        if not self.simulation_run:
            self.run_simulation()
        
        # Create figure with subplots arranged in a dashboard layout
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[2, 1, 1])
        
        # 3D visualization (top left, spanning 2 columns)
        ax_3d = fig.add_subplot(gs[0, :2], projection='3d')
        ax_3d.set_box_aspect([2, 1, 1])
        ax_3d.set_xlim(-self.rail_length * 0.2, self.rail_length * 1.2)
        ax_3d.set_ylim(-self.rail_separation * 3, self.rail_separation * 3)
        ax_3d.set_zlim(-self.rail_height * 2, self.rail_height * 6)
        ax_3d.set_xlabel('X (m)')
        ax_3d.set_ylabel('Y (m)')
        ax_3d.set_zlabel('Z (m)')
        ax_3d.set_title('Rail Gun 3D Model')
        
        # Add static components to 3D view
        self.add_static_components(ax_3d)
        
        # Performance metrics (top right)
        ax_metrics = fig.add_subplot(gs[0, 2])
        ax_metrics.axis('off')
        ax_metrics.set_title('Performance Metrics')
        
        # Position vs Time (middle left)
        ax_pos = fig.add_subplot(gs[1, 0])
        ax_pos.plot(self.time_points, self.positions, 'b-')
        ax_pos.axhline(y=self.rail_length, color='r', linestyle='--', label='Rail End')
        ax_pos.set_xlabel('Time (s)')
        ax_pos.set_ylabel('Position (m)')
        ax_pos.set_title('Projectile Position')
        ax_pos.grid(True)
        ax_pos.legend()
        
        # Velocity vs Time (middle center)
        ax_vel = fig.add_subplot(gs[1, 1])
        ax_vel.plot(self.time_points, self.velocities, 'g-')
        ax_vel.set_xlabel('Time (s)')
        ax_vel.set_ylabel('Velocity (m/s)')
        ax_vel.set_title('Projectile Velocity')
        ax_vel.grid(True)
        
        # Current vs Time (middle right)
        ax_curr = fig.add_subplot(gs[1, 2])
        ax_curr.plot(self.time_points, self.currents / 1000, 'r-')  # Display in kA
        ax_curr.set_xlabel('Time (s)')
        ax_curr.set_ylabel('Current (kA)')
        ax_curr.set_title('Circuit Current')
        ax_curr.grid(True)
        
        # Forces vs Time (bottom left)
        ax_force = fig.add_subplot(gs[2, 0])
        ax_force.plot(self.time_points, self.lorentz_forces / 1000, 'b-', label='Lorentz Force')
        ax_force.plot(self.time_points, self.drag_forces / 1000, 'r-', label='Drag Force')
        ax_force.set_xlabel('Time (s)')
        ax_force.set_ylabel('Force (kN)')
        ax_force.set_title('Forces')
        ax_force.grid(True)
        ax_force.legend()
        
        # Temperature vs Time (bottom center)
        ax_temp = fig.add_subplot(gs[2, 1])
        ax_temp.plot(self.time_points, self.temperatures - 273.15, 'orange')  # Convert to °C
        ax_temp.set_xlabel('Time (s)')
        ax_temp.set_ylabel('Temperature (°C)')
        ax_temp.set_title('Projectile Temperature')
        ax_temp.grid(True)
        
        # Energy vs Time (bottom right)
        ax_energy = fig.add_subplot(gs[2, 2])
        ax_energy.plot(self.time_points, self.energies / 1000, 'purple')  # Display in kJ
        ax_energy.set_xlabel('Time (s)')
        ax_energy.set_ylabel('Kinetic Energy (kJ)')
        ax_energy.set_title('Projectile Energy')
        ax_energy.grid(True)
        
        # Find where projectile exits the rails
        exit_indices = np.where(self.positions >= self.rail_length)[0]
        if len(exit_indices) > 0:
            exit_index = exit_indices[0]
            exit_time = self.time_points[exit_index]
            exit_velocity = self.velocities[exit_index]
            exit_energy = self.energies[exit_index] / 1000  # kJ
            exit_temperature = self.temperatures[exit_index] - 273.15  # °C
            
            # Add exit line to plots
            ax_pos.axvline(x=exit_time, color='k', linestyle='--', alpha=0.7)
            ax_vel.axvline(x=exit_time, color='k', linestyle='--', alpha=0.7)
            ax_curr.axvline(x=exit_time, color='k', linestyle='--', alpha=0.7)
            ax_force.axvline(x=exit_time, color='k', linestyle='--', alpha=0.7)
            ax_temp.axvline(x=exit_time, color='k', linestyle='--', alpha=0.7)
            ax_energy.axvline(x=exit_time, color='k', linestyle='--', alpha=0.7)
        else:
            exit_time = None
            exit_velocity = None
            exit_energy = None
            exit_temperature = None
        
        # Calculate efficiency
        final_energy = self.energies[-1]
        efficiency = (final_energy / self.capacitor_bank_energy) * 100
        
        # Add metrics text
        metrics_text = (
            f"Rail Length: {self.rail_length:.2f} m\n"
            f"Rail Separation: {self.rail_separation*1000:.1f} mm\n"
            f"Projectile Mass: {self.projectile_mass*1000:.1f} g\n\n"
            f"Exit Time: {exit_time:.3f} s\n" if exit_time else "Projectile did not exit\n"
            f"Exit Velocity: {exit_velocity:.1f} m/s\n" if exit_velocity else ""
            f"Peak Velocity: {np.max(self.velocities):.1f} m/s\n"
            f"Max Current: {np.max(self.currents)/1000:.1f} kA\n"
            f"Peak Force: {np.max(self.lorentz_forces)/1000:.1f} kN\n"
            f"Max Temperature: {np.max(self.temperatures)-273.15:.1f} °C\n"
            f"Energy Efficiency: {efficiency:.1f}%\n"
        )
        
        ax_metrics.text(0.05, 0.95, metrics_text, transform=ax_metrics.transAxes, 
                       verticalalignment='top', fontsize=10)
        
        # Initialize projectile in 3D view
        initial_projectile_faces = self.create_projectile_vertices(self.initial_position)
        projectile = Poly3DCollection(initial_projectile_faces, alpha=0.9, linewidth=1, edgecolor='k')
        projectile.set_facecolor('darkred')
        ax_3d.add_collection3d(projectile)
        
        # Add a time slider for controlling visualization
        ax_slider = plt.axes([0.2, 0.01, 0.65, 0.03])
        time_slider = Slider(
            ax=ax_slider,
            label='Time',
            valmin=0,
            valmax=self.time_points[-1],
            valinit=0,
            valfmt='%.3f s'
        )
        
        # Function to update visualization based on slider
        def update(val):
            # Get the time value from slider
            t = time_slider.val
            
            # Find closest time index
            time_index = np.argmin(np.abs(self.time_points - t))
            
            # Get values at selected time
            pos = self.positions[time_index]
            curr = self.currents[time_index]
            temp = self.temperatures[time_index]
            
            # Update projectile position and color
            new_projectile_faces = self.create_projectile_vertices(pos)
            projectile.set_verts(new_projectile_faces)
            
            # Update projectile color based on temperature
            temp_norm = min(1.0, max(0, (temp - self.ambient_temperature) / 1000))
            projectile.set_facecolor(plt.cm.plasma(temp_norm))
            
            # Update plots with time marker
            for ax in [ax_pos, ax_vel, ax_curr, ax_force, ax_temp, ax_energy]:
                # Remove old time markers
                for line in ax.get_lines():
                    if line.get_label() == '_time_marker':
                        line.remove()
                
                # Add new time marker
                ax.axvline(x=t, color='black', linestyle='-', linewidth=1, alpha=0.7, label='_time_marker')
            
            fig.canvas.draw_idle()
        
        # Connect the slider to the update function
        time_slider.on_changed(update)
        
        plt.tight_layout()
        return fig, time_slider
    
    def run_parameter_sweep(self, parameter, values, fixed_params=None):
        """
        Run a parameter sweep by varying one parameter and measuring results.
        
        Parameters:
        parameter (str): The parameter to vary
        values (list): List of values to test for the parameter
        fixed_params (dict): Dictionary of other parameters to set
        
        Returns:
        dict: Dictionary with results for each parameter value
        """
        # Store original parameters
        original_params = {
            'rail_length': self.rail_length,
            'rail_separation': self.rail_separation,
            'projectile_mass': self.projectile_mass,
            'capacitor_bank_energy': self.capacitor_bank_energy
        }
        
        # Setup result containers
        results = {
            'parameter_values': values,
            'exit_velocities': [],
            'exit_times': [],
            'max_currents': [],
            'max_temperatures': [],
            'efficiencies': []
        }
        
        # Apply fixed parameters if provided
        if fixed_params:
            for param, value in fixed_params.items():
                setattr(self, param, value)
        
        # Run simulation for each parameter value
        for value in values:
            # Set the parameter value
            setattr(self, parameter, value)
            
            # Reset simulation flags and run
            self.simulation_run = False
            sim_results = self.run_simulation()
            
            # Store results
            if sim_results['exit_velocity'] is not None:
                results['exit_velocities'].append(sim_results['exit_velocity'])
                results['exit_times'].append(sim_results['exit_time'])
            else:
                results['exit_velocities'].append(0)
                results['exit_times'].append(self.simulation_time)
                
            results['max_currents'].append(np.max(self.currents))
            results['max_temperatures'].append(np.max(self.temperatures) - 273.15)  # Convert to °C
            results['efficiencies'].append(sim_results['efficiency'])
        
        # Restore original parameters
        for param, value in original_params.items():
            setattr(self, param, value)
        
        return results
    
    def plot_parameter_sweep(self, parameter, results, parameter_label=None):
        """
        Plot the results of a parameter sweep.
        
        Parameters:
        parameter (str): The parameter that was varied
        results (dict): Results dictionary from run_parameter_sweep
        parameter_label (str): Optional label for the parameter
        
        Returns:
        matplotlib.figure.Figure: The figure with plots
        """
        if parameter_label is None:
            parameter_label = parameter.replace('_', ' ').title()
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Exit Velocity
        axs[0, 0].plot(results['parameter_values'], results['exit_velocities'], 'o-', color='blue')
        axs[0, 0].set_xlabel(parameter_label)
        axs[0, 0].set_ylabel('Exit Velocity (m/s)')
        axs[0, 0].set_title('Effect on Exit Velocity')
        axs[0, 0].grid(True)
        
        # Exit Time
        axs[0, 1].plot(results['parameter_values'], results['exit_times'], 'o-', color='green')
        axs[0, 1].set_xlabel(parameter_label)
        axs[0, 1].set_ylabel('Exit Time (s)')
        axs[0, 1].set_title('Effect on Exit Time')
        axs[0, 1].grid(True)
        
        # Max Current
        axs[1, 0].plot(results['parameter_values'], np.array(results['max_currents'])/1000, 'o-', color='red')
        axs[1, 0].set_xlabel(parameter_label)
        axs[1, 0].set_ylabel('Max Current (kA)')
        axs[1, 0].set_title('Effect on Maximum Current')
        axs[1, 0].grid(True)
        
        # Efficiency
        axs[1, 1].plot(results['parameter_values'], results['efficiencies'], 'o-', color='purple')
        axs[1, 1].set_xlabel(parameter_label)
        axs[1, 1].set_ylabel('Energy Efficiency (%)')
        axs[1, 1].set_title('Effect on Energy Efficiency')
        axs[1, 1].grid(True)
        
        plt.tight_layout()
        fig.suptitle(f'Parameter Sweep: Effect of {parameter_label}', fontsize=16)
        plt.subplots_adjust(top=0.92)
        
        return fig

# Example usage
if __name__ == "__main__":
    print("Initializing Advanced Rail Gun Simulation...")
    sim = AdvancedRailGunSimulation(
        # Physical parameters
        rail_length=5.0,          # meters
        rail_separation=0.15,     # meters
        projectile_mass=0.1,      # kg
        
        # Electrical parameters
        capacitor_bank_energy=500000,  # Joules (500 kJ)
        capacitor_voltage=5000,        # Volts
        
        # Simulation settings
        simulation_time=0.05,     # seconds
        time_steps=1000           # resolution
    )
    
    print("Running simulation...")
    results = sim.run_simulation()
    
    print(f"Rail Gun Simulation Results:")
    print(f"Final velocity: {results['velocity'][-1]:.2f} m/s")
    if results['exit_time'] is not None:
        print(f"Exit time: {results['exit_time']:.5f} s")
        print(f"Exit velocity: {results['exit_velocity']:.2f} m/s")
    else:
        print("Projectile did not exit the rails")
    print(f"Max current: {np.max(results['current'])/1000:.2f} kA")
    print(f"Max temperature: {np.max(results['temperature'])-273.15:.2f} °C")
    print(f"Energy efficiency: {results['efficiency']:.2f}%")
    
    print("\nCreating visualizations...")
    
    # Option 1: Create static 3D visualization
    print("Generating 3D model...")
    fig_3d, ax_3d = sim.visualize_static_3d()
    
    # Option 2: Create dashboard with plots and 3D view
    print("Generating dashboard...")
    fig_dashboard, slider = sim.create_dashboard()
    
    # Option 3: Create 3D animation
    print("Generating 3D animation (this may take a moment)...")
    animation = sim.animate_3d_simulation(save=False)
    
    # Option 4: Run parameter sweep (uncomment to use)
    # Vary projectile mass
    # masses = np.linspace(0.05, 0.3, 5)  # kg
    # mass_results = sim.run_parameter_sweep('projectile_mass', masses)
    # fig_mass = sim.plot_parameter_sweep('projectile_mass', mass_results, 'Projectile Mass (kg)')
    
    # Show plots
    plt.show(block=True)