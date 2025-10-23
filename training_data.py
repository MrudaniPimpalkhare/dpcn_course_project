import torch
import numpy as np
from torch.distributions import MultivariateNormal

# Set a default tensor type for performance
torch.set_default_dtype(torch.float32)

torch.set_default_device('cuda')

def build_correlation_matrix(N, K, cluster_assignments, intra_corrs, epsilon):
    """
    Builds the N x N correlation matrix for the synthetic market.

    Args:
        N (int): Total number of stocks.
        K (int): Total number of clusters.
        cluster_assignments (torch.Tensor): A tensor of shape (N,) mapping 
                                            each stock to a cluster index (0 to K-1).
        intra_corrs (torch.Tensor): A tensor of shape (K,) containing the 
                                    random correlation for each cluster.
        epsilon (float): The inter-cluster correlation value.

    Returns:
        torch.Tensor: The (N, N) correlation matrix.
    """
    # 1. Initialize the matrix with 1s on the diagonal
    corr_matrix = torch.eye(N)
    
    # 2. Use broadcasting to efficiently create a matrix of cluster IDs
    #    cluster_matrix[i, j] will be True if stock i and j are in the same cluster
    cluster_matrix = cluster_assignments.unsqueeze(0) == cluster_assignments.unsqueeze(1)
    
    # 3. Create a matrix of the intra-cluster correlation values
    #    We map the cluster-specific correlations to each stock
    intra_corr_map = intra_corrs[cluster_assignments]
    #    Now, we create an (N, N) matrix where intra_matrix[i, j] is the
    #    correlation for their cluster (if they are in the same cluster)
    intra_matrix = torch.min(intra_corr_map.unsqueeze(0), intra_corr_map.unsqueeze(1))
    
    # 4. Create the inter-cluster (epsilon) matrix
    #    This matrix has epsilon everywhere *except* the diagonal
    epsilon_matrix = torch.full((N, N), epsilon)
    
    # 5. Combine the matrices
    #    - Start with the epsilon matrix
    #    - Use torch.where to overwrite with intra-cluster values
    #      (if cluster_matrix is True, use intra_matrix, else use epsilon_matrix)
    corr_matrix = torch.where(cluster_matrix, intra_matrix, epsilon_matrix)
    
    # 6. Set the diagonal to 1.0 (self-correlation)
    corr_matrix.fill_diagonal_(1.0)

    print(f"Built correlation matrix with epsilon={epsilon:.3f}")
    print(corr_matrix)
    
    return corr_matrix

def generate_universe_data(N, K, T, w, epsilon_start=0.1, epsilon_end=0.9, epsilon_step=0.01):
    """
    Generates all training samples and the single tipping point label for 
    one "toy universe".

    Args:
        N (int): Total number of stocks for this universe.
        K (int): Total number of clusters for this universe.
        T (int): Time steps to generate at *each* epsilon step (e.g., 100).
        w (int): Sliding window size (e.g., 20).
        epsilon_start (float): The starting value for inter-cluster correlation.
        epsilon_end (float): The maximum value for inter-cluster correlation.
        epsilon_step (float): The increment for epsilon in each simulation step.

    Returns:
        (list of torch.Tensor, float): 
            - A list of all (N, w) training samples (or empty list if sim fails).
            - The single float label (epsilon_c) for this universe (or None).
    """
    
    # --- 1. Set up the "physics" for this unique universe ---
    
    # Assign each of the N stocks to one of K clusters
    cluster_assignments = torch.randint(0, K, (N,))
    
    # Assign a random "intra-cluster correlation" for each cluster
    # As requested, this is a random number between 0.5 and 1.0
    intra_corrs = (torch.rand(K) * 0.5) + 0.5

    all_time_series_data = []
    epsilon_steps = torch.arange(epsilon_start, epsilon_end + epsilon_step, epsilon_step)
    
    epsilon_c = None # This will be our label
    
    # --- 2. Run the simulation from "healthy" to "tipped" ---
    
    for i, epsilon_val in enumerate(epsilon_steps):
        epsilon_val = epsilon_val.item()
        
        # Build the correlation matrix for this epsilon step
        corr_matrix = build_correlation_matrix(N, K, cluster_assignments, intra_corrs, epsilon_val)
        
        # --- Find the "Tipping Point" (label) ---
        # We check if the matrix is still "valid" (positive-semidefinite).
        # A simple and robust check is to try a Cholesky decomposition.
        # If it fails, the matrix is "broken," and the system has "tipped."
        try:
            # A more stable check for positive-definiteness
            _ = torch.linalg.cholesky(corr_matrix)
        except RuntimeError:
            # Cholesky failed! The matrix is no longer positive-definite.
            # This is our critical transition.
            
            # The label (epsilon_c) is this "broken" value of epsilon
            epsilon_c = epsilon_val 
            
            # We stop the simulation. All data collected *so far*
            # is our pre-transition data.
            break
        
        # --- If system has NOT tipped, generate data ---
        
        # Define the multivariate normal distribution
        # `loc` is the mean (0 for returns), `covariance_matrix` is our matrix
        try:
            dist = MultivariateNormal(loc=torch.zeros(N), covariance_matrix=corr_matrix)
        except ValueError:
            # Catches other numerical instability.
            epsilon_c = epsilon_val
            break

        # Generate T time steps of stock returns
        # .sample() gives (T, N), so we transpose to (N, T)
        returns = dist.sample((T,)).T
        
        # Add the data for this epsilon step to our collection
        all_time_series_data.append(returns)
        
    # --- 3. Process the simulation results ---

    # If the loop finished without *ever* tipping, or tipped on the
    # very first step, this simulation is not useful.
    if epsilon_c is None or len(all_time_series_data) == 0:
        # print("Simulation failed to tip or tipped immediately.")
        return [], None
        
    # We have a valid simulation. Concatenate all pre-transition data
    # into one long time series: (N, total_pre_T_steps)
    full_pre_series = torch.cat(all_time_series_data, dim=1)
    total_len = full_pre_series.shape[1]
    
    # If the total data is shorter than our window, we can't make samples.
    if total_len < w:
        # print("Simulation tipped too early, not enough data for a window.")
        return [], None
        
    # --- 4. Create Training Samples using Sliding Window ---
    #
    
    samples = []
    
    # Slide the window of size `w` from t=0 to the end
    for t in range(total_len - w + 1):
        # Extract the window: shape (N, w)
        window = full_pre_series[:, t : t + w]
        samples.append(window)
        
    # Return all samples and the single label that applies to all of them
    #
    return samples, epsilon_c

def create_training_dataset(num_universes, T=100, w=20):
    """
    Creates the full training dataset by running many universe simulations.
    This is the "Data Generation" step 5 in your plan.

    Args:
        num_universes (int): The number of simulations to run (e.g., 1000).
        T (int): Time steps to generate at each epsilon step.
        w (int): Sliding window size.

    Returns:
        (list of torch.Tensor, list of float):
            - A "giant pool" of all (N, w) training samples.
            - A parallel list of corresponding labels (epsilon_c).
    """
    
    all_samples_in_pool = []
    all_labels_in_pool = []
    
    print(f"Starting data generation for {num_universes} universes...")
    
    for i in range(num_universes):
        # For each universe, randomize its "physics"
        #
        N_rand = np.random.randint(50, 201)  # Random N (50-200 stocks)
        K_rand = np.random.randint(3, 11)   # Random K (3-10 clusters)
        
        # Generate all samples and the one label for this universe
        samples, label = generate_universe_data(N=N_rand, K=K_rand, T=T, w=w)
        
        # --- 5. Pool the Data ---
        #
        if samples: # If the simulation was successful
            # Add all samples from this universe to the giant pool
            all_samples_in_pool.extend(samples)
            
            # Add the *same label* for each of those samples
            #
            all_labels_in_pool.extend([label] * len(samples))
            
        if (i + 1) % 100 == 0:
            print(f"  ...completed {i+1}/{num_universes} universes. "
                  f"Total samples so far: {len(all_samples_in_pool)}")

    print(f"--- Data generation complete. ---")
    print(f"Total samples generated: {len(all_samples_in_pool)}")
    print(f"Total labels generated:  {len(all_labels_in_pool)}")
    
    return all_samples_in_pool, all_labels_in_pool

# --- Example of how to run it ---
if __name__ == "__main__":
    
    # Run a small test (e.g., 10 universes)
    # For your real training, you'll set num_universes=1000+
    NUM_UNIVERSES = 10 
    WINDOW_SIZE = 20
    TIME_STEPS_PER_EPSILON = 100
    
    samples, labels = create_training_dataset(
        num_universes=NUM_UNIVERSES,
        T=TIME_STEPS_PER_EPSILON,
        w=WINDOW_SIZE
    )
    
    if samples:
        print("\n--- Example Output ---")
        print(f"Shape of first sample: {samples[0].shape}")
        print(f"Label for first sample: {labels[0]}")
        print(f"Shape of last sample: {samples[-1].shape}")
        print(f"Label for last sample: {labels[-1]}")
        
        # Verify that all samples from one universe get the same label
        # (if you run more than one simulation)