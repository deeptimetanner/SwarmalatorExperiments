import numpy as np
import matplotlib.pyplot as plt

class KuramotoModel:
    def __init__(self, n_oscillators, coupling_strength, dt=0.01):
        """
        Initialize Kuramoto model
        
        Args:
            n_oscillators (int): Number of oscillators
            coupling_strength (float): Coupling strength (K)
            dt (float): Time step for integration
        """
        self.n = n_oscillators
        self.K = coupling_strength
        self.dt = dt
        
        # Initialize natural frequencies from normal distribution with smaller variance
        self.natural_frequencies = np.random.normal(0, 0.1, self.n)
        
        # Initialize random phases between 0 and 2π
        self.phases = np.random.uniform(0, 2*np.pi, self.n)
        
    def calculate_phase_derivatives(self):
        """Calculate the derivatives of phases for all oscillators"""
        # Calculate sine of phase differences
        phase_differences = self.phases[np.newaxis, :] - self.phases[:, np.newaxis]
        interactions = np.sin(phase_differences)
        
        # Calculate phase derivatives
        dphases = self.natural_frequencies + (self.K / self.n) * np.sum(interactions, axis=1)
        return dphases
    
    def step(self):
        """Perform one time step using Euler method"""
        dphases = self.calculate_phase_derivatives()
        self.phases += self.dt * dphases
        
    def calculate_order_parameter(self):
        """Calculate the Kuramoto order parameter r"""
        complex_phases = np.exp(1j * self.phases)
        r = np.abs(np.mean(complex_phases))
        return r

def simulate_kuramoto(n_oscillators=100, coupling_strength=2.0, t_max=100, dt=0.01):
    """
    Simulate Kuramoto model and plot results
    
    Args:
        n_oscillators (int): Number of oscillators
        coupling_strength (float): Coupling strength (K)
        t_max (float): Maximum simulation time
        dt (float): Time step
    """
    # Initialize model
    model = KuramotoModel(n_oscillators, coupling_strength, dt)
    
    # Prepare arrays for storing results
    t_steps = int(t_max / dt)
    times = np.arange(t_steps) * dt
    order_parameters = np.zeros(t_steps)
    
    # Run simulation
    for i in range(t_steps):
        order_parameters[i] = model.calculate_order_parameter()
        model.step()
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(times, order_parameters)
    plt.xlabel('Time')
    plt.ylabel('Order Parameter (r)')
    plt.title('Kuramoto Model Synchronization')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Example usage with adjusted parameters
    simulate_kuramoto(n_oscillators=100, coupling_strength=5.0, t_max=100)