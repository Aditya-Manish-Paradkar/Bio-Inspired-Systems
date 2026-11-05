import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define the Problem (Mathematical function to minimize)
def objective_function(x):
    return np.sum(x**2)  # Sphere function (minimum at x = 0)

# Step 2: Initialize Parameters
num_wolves = 5         # Number of wolves in the pack
num_dimensions = 2     # Number of dimensions (for the optimization problem)
num_iterations = 30    # Number of iterations
lb = -10               # Lower bound of search space
ub = 10                # Upper bound of search space

# Step 3: Initialize Population (Generate initial positions randomly)
wolves = np.random.uniform(lb, ub, (num_wolves, num_dimensions))

# Initialize Alpha, Beta, Delta wolves' positions and scores
alpha_pos = np.zeros(num_dimensions)
beta_pos = np.zeros(num_dimensions)
delta_pos = np.zeros(num_dimensions)

alpha_score = float('inf')  # Best (Alpha) score
beta_score = float('inf')   # Second best (Beta) score
delta_score = float('inf')  # Third best (Delta) score

# Array to store the alpha score over iterations for plotting
alpha_score_history = []

# Step 4: Evaluate Fitness and assign Alpha, Beta, Delta wolves
def evaluate_fitness():
    global alpha_pos, beta_pos, delta_pos, alpha_score, beta_score, delta_score
    for wolf in wolves:
        fitness = objective_function(wolf)
        
        # Update Alpha, Beta, and Delta wolves based on fitness
        if fitness < alpha_score:
            delta_score = beta_score
            delta_pos = beta_pos.copy()
            beta_score = alpha_score
            beta_pos = alpha_pos.copy()
            alpha_score = fitness
            alpha_pos = wolf.copy()
        elif fitness < beta_score:
            delta_score = beta_score
            delta_pos = beta_pos.copy()
            beta_score = fitness
            beta_pos = wolf.copy()
        elif fitness < delta_score:
            delta_score = fitness
            delta_pos = wolf.copy()

# Step 5: Update Positions
def update_positions(iteration):
    # Parameter 'a' decreases linearly from 2 to 0 over iterations
    a = 2 - iteration * (2 / num_iterations)
    
    for i in range(num_wolves):
        for j in range(num_dimensions):
            # Position update based on Alpha wolf
            r1, r2 = np.random.random(), np.random.random()
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = abs(C1 * alpha_pos[j] - wolves[i, j])
            X1 = alpha_pos[j] - A1 * D_alpha

            # Position update based on Beta wolf
            r1, r2 = np.random.random(), np.random.random()
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = abs(C2 * beta_pos[j] - wolves[i, j])
            X2 = beta_pos[j] - A2 * D_beta

            # Position update based on Delta wolf
            r1, r2 = np.random.random(), np.random.random()
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = abs(C3 * delta_pos[j] - wolves[i, j])
            X3 = delta_pos[j] - A3 * D_delta

            # Update wolf position (average of X1, X2, X3)
            wolves[i, j] = (X1 + X2 + X3) / 3
            
            # Apply boundary constraints
            wolves[i, j] = np.clip(wolves[i, j], lb, ub)

# Step 6: Main GWO Optimization Loop
if __name__ == "__main__":
    for iteration in range(num_iterations):
        evaluate_fitness()           # Evaluate fitness of each wolf
        update_positions(iteration)   # Update positions based on Alpha, Beta, Delta
        
        # Record the alpha score for this iteration
        alpha_score_history.append(alpha_score)
        print(f"Iteration {iteration + 1:2d}/{num_iterations}, Alpha Score: {alpha_score:.6e}")

    # Step 7: Output the Best Solution
    print("\nBest Solution Position:", alpha_pos)
    print(f"Best Solution Fitness: {alpha_score:.6e}")

    # Plotting the convergence graph
    plt.figure(figsize=(8, 5))
    plt.plot(alpha_score_history, color='purple', linewidth=2)
    plt.title('Convergence of Grey Wolf Optimizer')
    plt.xlabel('Iteration')
    plt.ylabel('Alpha Fitness Score')
    plt.grid(True)
    plt.show()
