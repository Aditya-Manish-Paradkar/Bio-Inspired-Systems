import numpy as np
import random
import math
import matplotlib.pyplot as plt

# Define a sample function to optimize (Sphere function: f(x) = sum(x_i^2))
def objective_function(x):
    return np.sum(x ** 2)

# Lévy flight step generator
def levy_flight(Lambda):
    sigma_u = (
        math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
        (math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))
    ) ** (1 / Lambda)
    sigma_v = 1.0
    u = np.random.normal(0, sigma_u, size=1)
    v = np.random.normal(0, sigma_v, size=1)
    step = u / (np.abs(v) ** (1 / Lambda))
    return step

# Cuckoo Search algorithm
def cuckoo_search(num_nests=25, num_iterations=100, discovery_rate=0.25, dim=5, 
                  lower_bound=-10, upper_bound=10):
    # Initialize nests randomly within search boundaries
    nests = np.random.uniform(lower_bound, upper_bound, (num_nests, dim))
    fitness = np.array([objective_function(nest) for nest in nests])

    # Find initial global best nest
    best_nest_idx = np.argmin(fitness)
    best_nest = nests[best_nest_idx].copy()
    best_fitness = fitness[best_nest_idx]

    Lambda = 1.5  # Parameter for Lévy flights
    fitness_history = []  # To track best fitness per iteration

    for iteration in range(num_iterations):
        # 1. Generate new solutions via Lévy flight step
        for i in range(num_nests):
            step_size = levy_flight(Lambda)
            new_solution = nests[i] + step_size * (nests[i] - best_nest)
            new_solution = np.clip(new_solution, lower_bound, upper_bound)
            new_fitness = objective_function(new_solution)

            # Greedy replacement if new solution is better
            if new_fitness < fitness[i]:
                nests[i] = new_solution
                fitness[i] = new_fitness

        # 2. Abandon a fraction (discovery_rate) of worse nests
        random_nests = np.random.choice(num_nests, int(discovery_rate * num_nests), replace=False)
        for nest_idx in random_nests:
            nests[nest_idx] = np.random.uniform(lower_bound, upper_bound, dim)
            fitness[nest_idx] = objective_function(nests[nest_idx])

        # 3. Update the global best solution
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_fitness = fitness[current_best_idx]
            best_nest = nests[current_best_idx].copy()

        # Track fitness history
        fitness_history.append(best_fitness)

        # Print progress every iteration
        print(f"Iteration {iteration + 1:2d}/{num_iterations}, Best Fitness: {best_fitness:.6e}")

    # Plot fitness convergence graph
    plt.figure(figsize=(8, 5))
    plt.plot(fitness_history, color='b', linewidth=2)
    plt.title('Fitness Convergence Over Iterations (Cuckoo Search)')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.grid(True)
    plt.show()

    return best_nest, best_fitness

# Example execution
if __name__ == "__main__":
    best_nest, best_fitness = cuckoo_search(
        num_nests=30, 
        num_iterations=30, 
        dim=10, 
        lower_bound=-5, 
        upper_bound=5
    )
    
    print("\nBest Solution Vector Found:")
    print(best_nest)
    print(f"\nBest Fitness Value: {best_fitness:.6e}")
