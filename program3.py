import numpy as np
import matplotlib.pyplot as plt

# 1. Define the Problem: Create a set of cities with their coordinates
cities = np.array([
    [0, 0], # City 0
    [1, 5], # City 1
    [5, 1], # City 2
    [6, 4], # City 3
    [7, 8], # City 4
])

# Calculate the distance matrix between each pair of cities
def calculate_distances(cities):
    num_cities = len(cities)
    distances = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            distances[i][j] = np.linalg.norm(cities[i] - cities[j])
    return distances

distances = calculate_distances(cities)

# 2. Initialize Parameters
num_ants = 10
num_cities = len(cities)
alpha = 1.0  # Influence of pheromone
beta = 5.0   # Influence of heuristic (inverse distance)
rho = 0.5    # Evaporation rate
num_iterations = 30
initial_pheromone = 1.0

# 3. Heuristic information (Inverse of distance)
def heuristic(distances):
    with np.errstate(divide='ignore'):  # Ignore division by zero on diagonal
        eta_matrix = 1.0 / distances
        np.fill_diagonal(eta_matrix, 0) # Set self-distance heuristic to zero
    return eta_matrix

eta = heuristic(distances)

# 4. Choose next city probabilistically based on pheromone and heuristic info
def choose_next_city(pheromone, eta, visited):
    probs = []
    current_city = visited[-1]
    
    for j in range(num_cities):
        if j not in visited:
            pheromone_ij = pheromone[current_city, j] ** alpha
            heuristic_ij = eta[current_city, j] ** beta
            probs.append(pheromone_ij * heuristic_ij)
        else:
            probs.append(0.0)
            
    probs = np.array(probs)
    prob_sum = probs.sum()
    
    # Fallback to uniform random choice among unvisited cities if probability sum is 0
    if prob_sum == 0:
        unvisited = [j for j in range(num_cities) if j not in visited]
        return np.random.choice(unvisited)
        
    return np.random.choice(range(num_cities), p=probs / prob_sum)

# Construct solution for a single ant
def construct_solution(pheromone, eta):
    tour = [np.random.randint(0, num_cities)]
    while len(tour) < num_cities:
        next_city = choose_next_city(pheromone, eta, tour)
        tour.append(next_city)
    return tour

# 5. Update pheromones after all ants have constructed their tours
def update_pheromones(pheromone, all_tours, distances, best_tour):
    pheromone *= (1 - rho)  # Evaporate pheromones
    
    # Add pheromones for each ant's tour
    for tour in all_tours:
        tour_length = sum([distances[tour[i], tour[(i + 1) % num_cities]] for i in range(num_cities)])
        for i in range(num_cities):
            u, v = tour[i], tour[(i + 1) % num_cities]
            pheromone[u, v] += 1.0 / tour_length
            pheromone[v, u] += 1.0 / tour_length  # Symmetric graph update

    # Extra reinforcement on the iteration best tour
    if best_tour is not None:
        best_length = sum([distances[best_tour[i], best_tour[(i + 1) % num_cities]] for i in range(num_cities)])
        for i in range(num_cities):
            u, v = best_tour[i], best_tour[(i + 1) % num_cities]
            pheromone[u, v] += 1.0 / best_length
            pheromone[v, u] += 1.0 / best_length

# 6. Main ACO Loop: Iterate over multiple iterations to find the best solution
def run_aco(distances, num_iterations):
    pheromone = np.ones((num_cities, num_cities)) * initial_pheromone
    best_tour = None
    best_length = float('inf')
    
    for iteration in range(num_iterations):
        all_tours = [construct_solution(pheromone, eta) for _ in range(num_ants)]
        all_lengths = [
            sum([distances[tour[i], tour[(i + 1) % num_cities]] for i in range(num_cities)]) 
            for tour in all_tours
        ]

        current_best_length = min(all_lengths)
        current_best_tour = all_tours[all_lengths.index(current_best_length)]

        if current_best_length < best_length:
            best_length = current_best_length
            best_tour = current_best_tour

        update_pheromones(pheromone, all_tours, distances, best_tour)
        print(f"Iteration {iteration + 1:2d}, Best Length: {best_length:.4f}")

    return best_tour, best_length

# 8. Plot the Best Route
def plot_route(cities, best_tour, best_length):
    plt.figure(figsize=(8, 6))
    for i in range(len(cities)):
        plt.scatter(cities[i][0], cities[i][1], color='red', s=100, zorder=3)
        plt.text(cities[i][0] + 0.1, cities[i][1] + 0.1, f"City {i}", fontsize=12)

    # Plot the tour as lines connecting the cities (closing the loop back to start)
    tour_cities = np.array([cities[i] for i in best_tour] + [cities[best_tour[0]]])
    plt.plot(tour_cities[:, 0], tour_cities[:, 1], linestyle='-', marker='o', color='blue', zorder=2)
    plt.title(f"Best Tour (Length: {best_length:.4f})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()

# Run the ACO algorithm and output results
if __name__ == "__main__":
    best_tour, best_length = run_aco(distances, num_iterations)
    
    print("\nBest Solution Found:")
    print(f"Best Tour: {best_tour}")
    print(f"Best Tour Length: {best_length:.4f}")
    
    plot_route(cities, best_tour, best_length)
