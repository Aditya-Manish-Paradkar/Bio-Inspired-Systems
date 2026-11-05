import numpy as np
import matplotlib.pyplot as plt

# Define a sample function to optimize (Sphere function: f(x) = sum(x_i^2))
def objective_function(x):
    return np.sum(x ** 2)

# Gene expression: translate a binary genetic sequence (genotype) into real values (phenotype)
def express_gene(chromosome, dim, bits_per_gene, lower_bound, upper_bound):
    genes = chromosome.reshape(dim, bits_per_gene)
    weights = 2 ** np.arange(bits_per_gene - 1, -1, -1)
    integer_values = genes @ weights
    max_value = 2 ** bits_per_gene - 1
    return lower_bound + integer_values / max_value * (upper_bound - lower_bound)

# Selection: tournament selection based on fitness
def tournament_selection(population, fitness, tournament_size=3):
    selected = []
    for _ in range(len(population)):
        contenders = np.random.choice(len(population), tournament_size, replace=False)
        winner = contenders[np.argmin(fitness[contenders])]
        selected.append(population[winner].copy())
    return np.array(selected)

# Crossover: single-point crossover between pairs of parents
def crossover(parents, crossover_rate):
    offspring = parents.copy()
    chrom_len = parents.shape[1]
    for i in range(0, len(parents) - 1, 2):
        if np.random.rand() < crossover_rate:
            point = np.random.randint(1, chrom_len)
            offspring[i, point:] = parents[i + 1, point:]
            offspring[i + 1, point:] = parents[i, point:]
    return offspring

# Mutation: bit-flip mutation
def mutation(offspring, mutation_rate):
    flip_mask = np.random.rand(*offspring.shape) < mutation_rate
    offspring[flip_mask] = 1 - offspring[flip_mask]
    return offspring

# Gene Expression Algorithm
def gene_expression_algorithm(population_size=50, num_generations=100, dim=5,
                              bits_per_gene=16, mutation_rate=0.01, crossover_rate=0.8,
                              lower_bound=-10, upper_bound=10):
    chrom_len = dim * bits_per_gene

    # Initialize population of random genetic sequences (binary)
    population = np.random.randint(0, 2, (population_size, chrom_len))

    # Evaluate initial fitness (after gene expression)
    def evaluate(pop):
        return np.array([objective_function(
            express_gene(c, dim, bits_per_gene, lower_bound, upper_bound)) for c in pop])

    fitness = evaluate(population)

    # Find initial global best
    best_idx = np.argmin(fitness)
    best_chromosome = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    fitness_history = []  # To track best fitness per generation

    for generation in range(num_generations):
        # 1. Selection
        parents = tournament_selection(population, fitness)

        # 2. Crossover
        offspring = crossover(parents, crossover_rate)

        # 3. Mutation
        offspring = mutation(offspring, mutation_rate)

        # 4. Elitism: keep the best sequence found so far
        offspring[0] = best_chromosome

        # 5. Gene expression + fitness evaluation
        population = offspring
        fitness = evaluate(population)

        # 6. Update the global best solution
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_fitness = fitness[current_best_idx]
            best_chromosome = population[current_best_idx].copy()

        # Track fitness history
        fitness_history.append(best_fitness)

        # Print progress every generation
        print(f"Generation {generation + 1:3d}/{num_generations}, Best Fitness: {best_fitness:.6e}")

    # Plot fitness convergence graph
    plt.figure(figsize=(8, 5))
    plt.plot(fitness_history, color='r', linewidth=2)
    plt.title('Fitness Convergence Over Generations (Gene Expression Algorithm)')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.grid(True)
    plt.savefig('gea_convergence.png', dpi=120)
    plt.show()

    best_solution = express_gene(best_chromosome, dim, bits_per_gene, lower_bound, upper_bound)
    return best_solution, best_fitness

# Example execution
if __name__ == "__main__":
    best_solution, best_fitness = gene_expression_algorithm(
        population_size=50,
        num_generations=100,
        dim=10,
        bits_per_gene=16,
        mutation_rate=0.01,
        crossover_rate=0.8,
        lower_bound=-5,
        upper_bound=5
    )

    print("\nBest Solution Vector Found:")
    print(best_solution)
    print(f"\nBest Fitness Value: {best_fitness:.6e}")
