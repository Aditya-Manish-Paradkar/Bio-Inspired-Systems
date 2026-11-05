import numpy as np

# Define the optimization problem (Rastrigin Function)
def rastrigin(x):
    A = 10
    return A * len(x) + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

# Particle Swarm Optimization (PSO) implementation
class Particle:
    def __init__(self, dimension, lower_bound, upper_bound):
        # Initialize the particle position and velocity randomly
        self.position = np.random.uniform(lower_bound, upper_bound, dimension)
        self.velocity = np.random.uniform(-1, 1, dimension)
        self.best_position = np.copy(self.position)
        self.best_value = rastrigin(self.position)

    def update_velocity(self, global_best_position, w, c1, c2):
        # Update the velocity of the particle
        r1 = np.random.rand(len(self.position))
        r2 = np.random.rand(len(self.position))
        # Inertia term
        inertia = w * self.velocity
        # Cognitive term (individual best)
        cognitive = c1 * r1 * (self.best_position - self.position)
        # Social term (global best)
        social = c2 * r2 * (global_best_position - self.position)
        # Update velocity
        self.velocity = inertia + cognitive + social

    def update_position(self, lower_bound, upper_bound):
        # Update the position of the particle
        self.position = self.position + self.velocity
        # Ensure the particle stays within the bounds
        self.position = np.clip(self.position, lower_bound, upper_bound)

    def evaluate(self):
        # Evaluate the fitness of the particle
        fitness = rastrigin(self.position)
        # Update the particle's best position if necessary
        if fitness < self.best_value:
            self.best_value = fitness
            self.best_position = np.copy(self.position)

def particle_swarm_optimization(dim, lower_bound, upper_bound, num_particles=30, max_iter=100,
                               w=0.5, c1=1.5, c2=1.5):
    # Initialize particles
    particles = [Particle(dim, lower_bound, upper_bound) for _ in range(num_particles)]
    
    # Initialize the global best position and value
    global_best_position = np.copy(particles[0].best_position)
    global_best_value = particles[0].best_value
    
    # Find initial global best among all particles
    for particle in particles:
        if particle.best_value < global_best_value:
            global_best_value = particle.best_value
            global_best_position = np.copy(particle.best_position)

    for i in range(max_iter):
        # Update each particle
        for particle in particles:
            particle.update_velocity(global_best_position, w, c1, c2)
            particle.update_position(lower_bound, upper_bound)
            particle.evaluate()
            
            # Update global best position if needed
            if particle.best_value < global_best_value:
                global_best_value = particle.best_value
                global_best_position = np.copy(particle.best_position)
                
        # Print progress every 10 iterations
        if (i + 1) % 10 == 0:
            print(f"Iteration {i + 1}/{max_iter} - Best Fitness: {global_best_value:.6f}")
            
    return global_best_position, global_best_value

# Run PSO algorithm
if __name__ == "__main__":
    # Parameters for the PSO algorithm
    dim = 2             # Number of dimensions
    lower_bound = -5.12 # Lower bound of search space
    upper_bound = 5.12  # Upper bound of search space
    num_particles = 30  # Number of particles in swarm
    max_iter = 100      # Maximum iterations

    best_position, best_value = particle_swarm_optimization(
        dim, lower_bound, upper_bound, num_particles, max_iter
    )

    print("\nBest Solution Found:")
    print("Position:", best_position)
    print(f"Fitness: {best_value:.6f}")
