import numpy as np
import matplotlib.pyplot as plt

# Define a sample function to optimize (Sphere function: f(x) = sum(x_i^2))
def objective_function(x):
    return np.sum(x ** 2)

# Moore neighbourhood offsets (8 neighbours)
NEIGHBOURHOOD = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),           (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]

# Parallel Cellular Algorithm
def parallel_cellular_algorithm(grid_rows=6, grid_cols=5, num_iterations=100, dim=5,
                                lower_bound=-10, upper_bound=10, noise_scale=0.1):
    # Initialize cells randomly within search boundaries (each cell = one solution)
    cells = np.random.uniform(lower_bound, upper_bound, (grid_rows, grid_cols, dim))
    fitness = np.apply_along_axis(objective_function, 2, cells)

    # Find initial global best cell
    flat_idx = np.argmin(fitness)
    r0, c0 = divmod(flat_idx, grid_cols)
    best_cell = cells[r0, c0].copy()
    best_fitness = fitness[r0, c0]

    fitness_history = []  # To track best fitness per iteration

    for iteration in range(num_iterations):
        # 1. Find the best neighbour of every cell (toroidal grid, all cells at once)
        best_nb_fit = np.full((grid_rows, grid_cols), np.inf)
        best_nb_pos = np.zeros_like(cells)
        for dr, dc in NEIGHBOURHOOD:
            nb_fit = np.roll(fitness, shift=(-dr, -dc), axis=(0, 1))
            nb_pos = np.roll(cells, shift=(-dr, -dc), axis=(0, 1))
            better = nb_fit < best_nb_fit
            best_nb_fit = np.where(better, nb_fit, best_nb_fit)
            best_nb_pos = np.where(better[:, :, None], nb_pos, best_nb_pos)

        # 2. Synchronous state update: move toward best neighbour + decaying noise
        step = np.random.rand(grid_rows, grid_cols, 1)
        sigma = noise_scale * (upper_bound - lower_bound) * (1 - iteration / num_iterations) + 1e-9
        new_cells = cells + step * (best_nb_pos - cells) \
                    + np.random.normal(0, sigma, cells.shape)
        new_cells = np.clip(new_cells, lower_bound, upper_bound)
        new_fitness = np.apply_along_axis(objective_function, 2, new_cells)

        # Greedy replacement if new state is better
        improved = new_fitness < fitness
        cells[improved] = new_cells[improved]
        fitness[improved] = new_fitness[improved]

        # 3. Update the global best solution
        flat_idx = np.argmin(fitness)
        r, c = divmod(flat_idx, grid_cols)
        if fitness[r, c] < best_fitness:
            best_fitness = fitness[r, c]
            best_cell = cells[r, c].copy()

        # Track fitness history
        fitness_history.append(best_fitness)

        # Print progress every iteration
        print(f"Iteration {iteration + 1:2d}/{num_iterations}, Best Fitness: {best_fitness:.6e}")

    # Plot fitness convergence graph
    plt.figure(figsize=(8, 5))
    plt.plot(fitness_history, color='g', linewidth=2)
    plt.title('Fitness Convergence Over Iterations (Parallel Cellular Algorithm)')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.grid(True)
    plt.savefig('pca_convergence.png', dpi=120)
    plt.show()

    return best_cell, best_fitness

# Example execution
if __name__ == "__main__":
    best_cell, best_fitness = parallel_cellular_algorithm(
        grid_rows=6,
        grid_cols=5,
        num_iterations=30,
        dim=10,
        lower_bound=-5,
        upper_bound=5
    )

    print("\nBest Solution Vector Found:")
    print(best_cell)
    print(f"\nBest Fitness Value: {best_fitness:.6e}")
