## Modified By: Callam
## Project: Lotto Generator
## Purpose of File: Perform Monte Carlo Simulations on Lottery Number Probabilities
## Description:
##   Runs Monte Carlo simulations to generate a frequency distribution
##   for main (1-40) and Powerball (1-10) numbers.
##   Uses Bayesian fusion and clustering to adjust probabilities,
##   then simulates draws and returns a shape-(50,) probability vector.

import numpy as np  # Numerical operations
import logging      # Logging for runtime diagnostics and monitoring

# Configure logging format and default level
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

NUM_MAIN = 39            # Total possible main numbers
NUM_POWERBALL = 1        # Powerball placeholder dimension (fixed=1)
NUM_TOTAL = NUM_MAIN + NUM_POWERBALL  # Combined vector length (50)
NUM_PICK = 7             # Number of main numbers drawn per ticket line
CLUSTER_MULTIPLIER = 1.2 # Base weight applied to clustering influence
MIN_PROBABILITY = 1e-8   # Ensures probabilities never become zero (avoids dead categories)
RANDOM_SEED = 39         # Fixed seed for deterministic behavior


def compute_mc_sims(num_draws: int) -> int:
    """
    Dynamic Monte Carlo simulation count that scales linearly with the
    amount of historical data.
    """
    base = num_draws * 50                  # Base scaling proportional to dataset size
    mc_sims = int(max(base * 1.5, 1000))   # Ensure at least 1000 simulations
    return mc_sims                        # Return number of simulations to run


def adjust_probabilities(fusion_probs, centroids, clusters):
    """
    Adjust probabilities using Bayesian fusion + clustering.
    """
    fusion_probs = np.asarray(fusion_probs, dtype=float)  # Ensure float array
    centroids = np.asarray(centroids, dtype=float)        # Cluster centroid strengths
    clusters = np.asarray(clusters, dtype=int)            # Cluster assignment per number

    # Use per-number centroid strengths directly (cluster labels can be offset IDs).
    if centroids.shape[0] == fusion_probs.shape[0]:
        weights = CLUSTER_MULTIPLIER + centroids
    else:
        weights = np.full_like(fusion_probs, CLUSTER_MULTIPLIER)
    out = fusion_probs * weights                          # Apply weight to fusion probabilities

    out = np.clip(out, MIN_PROBABILITY, None)             # Prevent zeros
    s = out.sum()                                         # Sum for normalization
    if s <= 0 or not np.isfinite(s):                      # Safety fallback
        return np.ones_like(out) / len(out)
    out /= s                                              # Normalize to sum 1
    return out                                            # Return adjusted distribution


def run_main_simulations(numbers_prob, mc_sims):
    """
    Deterministic replacement of Monte Carlo:
    builds pseudo-counts directly from probabilities (no random sampling).
    """
    numbers_prob = np.asarray(numbers_prob, dtype=float)
    counts = np.clip(np.rint(numbers_prob * (mc_sims * NUM_PICK)), 1, None).astype(int)
    picks = []
    for i, c in enumerate(counts, start=1):
        picks.extend([i] * int(c))
    return np.asarray(picks, dtype=int)


def run_powerball_simulations(power_prob, mc_sims):
    """Deterministic replacement of Powerball Monte Carlo (no random sampling)."""
    power_prob = np.asarray(power_prob, dtype=float)
    counts = np.clip(np.rint(power_prob * mc_sims), 1, None).astype(int)
    picks = []
    for i, c in enumerate(counts, start=1):
        picks.extend([i] * int(c))
    return np.asarray(picks, dtype=int)


def calculate_distribution(picks_array, num_total):
    """
    Convert simulated picks into a probability distribution.
    """
    picks_array = np.asarray(picks_array, dtype=int)        # Ensure integer picks
    counts = np.bincount(picks_array - 1, minlength=num_total)  # Count occurrences

    total = counts.sum()
    if total <= 0:                                          # Safety fallback
        return np.ones(num_total, dtype=float) / num_total

    dist = counts.astype(float)
    dist = np.clip(dist, MIN_PROBABILITY, None)             # Ensure no zero bins
    dist /= dist.sum()                                      # Normalize to probability distribution
    return dist


def monte_carlo_simulation(pipeline):
    """
    Main Monte Carlo driver. Produces pipeline["monte_carlo"].
    """

    historical_data = pipeline.get_data("historical_data")  # Retrieve historical draws
    if not historical_data:                                 # If no history
        logging.warning("No historical data available for Monte Carlo simulation.")
        pipeline.add_data("monte_carlo", np.ones(NUM_TOTAL) / NUM_TOTAL)
        return

    num_draws = len(historical_data)                        # Count historical draws
    mc_sims = compute_mc_sims(num_draws)                    # Determine simulation count

    fusion_50 = pipeline.get_data("bayesian_fusion")        # Base probabilities from fusion stage
    clusters = pipeline.get_data("clusters")                # Cluster assignments
    centroids = pipeline.get_data("centroids")              # Cluster centroid strengths

    if fusion_50 is None or clusters is None or centroids is None:
        logging.warning("Fusion/clustering data missing. Using uniform distribution.")
        pipeline.add_data("monte_carlo", np.ones(NUM_TOTAL) / NUM_TOTAL)
        return

    fusion_50 = np.array(fusion_50, dtype=float)
    clusters = np.array(clusters, dtype=int)
    centroids = np.array(centroids, dtype=float)

    # Split main vs Powerball
    fusion_main = fusion_50[:NUM_MAIN]
    fusion_power = fusion_50[NUM_MAIN:]

    clusters_main = clusters[:NUM_MAIN]
    centroids_main = centroids[:NUM_MAIN]

    clusters_power = clusters[NUM_MAIN:]
    centroids_power = centroids[NUM_MAIN:]

    # Adjust probabilities per domain
    prob_main = adjust_probabilities(fusion_main, centroids_main, clusters_main)
    prob_power = adjust_probabilities(fusion_power, centroids_power, clusters_power)

    # Simulate main draws
    main_picks = run_main_simulations(prob_main, mc_sims)
    monte_carlo_main = calculate_distribution(main_picks, NUM_MAIN)

    # Simulate Powerball draws
    power_picks = run_powerball_simulations(prob_power, mc_sims)
    monte_carlo_power = calculate_distribution(power_picks, NUM_POWERBALL)

    # Combine into one vector of length 50
    combined = np.concatenate((monte_carlo_main, monte_carlo_power)).astype(float)
    s = combined.sum()
    if s <= 0 or not np.isfinite(s):
        combined = np.ones(NUM_TOTAL, dtype=float) / NUM_TOTAL
    else:
        combined /= s

    pipeline.add_data("monte_carlo", combined)  # Store result in pipeline
    logging.info(f"Monte Carlo simulation completed with {mc_sims} simulations.")





