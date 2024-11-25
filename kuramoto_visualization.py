import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from kuramoto_model import KuramotoModel

def animate_kuramoto(n_oscillators=100, coupling_strength=4.0, t_max=50, dt=0.01):
    """
    Animate Kuramoto model showing oscillators on unit circle and order parameter
    
    Args:
        n_oscillators (int): Number of oscillators
        coupling_strength (float): Coupling strength (K)
        t_max (float): Maximum simulation time
        dt (float): Time step
    """
    model = KuramotoModel(n_oscillators, coupling_strength, dt)
    
    # Set up the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Setup circle plot
    circle = plt.Circle((0, 0), 1, fill=False, color='black')
    ax1.add_artist(circle)
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.set_title('Oscillators on Unit Circle')
    
    # Setup order parameter plot
    t_steps = int(t_max / dt)
    times = np.arange(t_steps) * dt
    order_parameters = np.zeros(t_steps)
    line2, = ax2.plot([], [])
    ax2.set_xlim(0, t_max)
    ax2.set_ylim(0, 1)
    ax2.grid(True)
    ax2.set_title('Order Parameter Over Time')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Order Parameter (r)')
    
    # Initialize scatter plot for oscillators
    scatter = ax1.scatter([], [])
    
    # Animation update function
    def update(frame):
        # Update model
        order_parameters[frame] = model.calculate_order_parameter()
        model.step()
        
        # Calculate positions on unit circle
        x = np.cos(model.phases)
        y = np.sin(model.phases)
        
        # Update scatter plot
        scatter.set_offsets(np.c_[x, y])
        
        # Update order parameter plot
        line2.set_data(times[:frame+1], order_parameters[:frame+1])
        
        return scatter, line2

    # Create animation with larger interval (less frequent updates)
    anim = FuncAnimation(fig, update, frames=t_steps, 
                        interval=5,  # Reduced from 20 to 5 milliseconds
                        blit=True)
    
    plt.tight_layout()
    plt.show()
    
    # Optionally save animation
    # anim.save('kuramoto.gif', writer='pillow')

if __name__ == "__main__":
    # Example usage with stronger coupling and longer simulation time
    animate_kuramoto(
        n_oscillators=50, 
        coupling_strength=5.0,  # Increased from 1.0 to 5.0 for stronger coupling
        t_max=100,             # Increased from 50 to 100 to see full synchronization
        dt=0.05
    )
