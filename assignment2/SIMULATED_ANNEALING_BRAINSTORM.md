# Simulated Annealing for Base Station Placement Optimization

## Problem Statement

**Goal**: Find optimal locations for `NUM_BASE_STATIONS` base stations on the USC campus map to maximize the percentage of outdoor area covered.

**Current Approach**: Random placement (baseline)

**Proposed Approach**: Simulated Annealing (SA) - a metaheuristic optimization algorithm that can escape local optima.

---

## 1. Core Concepts of Simulated Annealing

### How It Works
1. **Start** with a random initial solution (random base station locations)
2. **Iterate**:
   - Generate a "neighbor" solution (slightly modify current solution)
   - Calculate the "energy" (cost) of both solutions
   - Accept the neighbor if it's better
   - Accept the neighbor even if it's worse, with probability based on temperature
3. **Cool down**: Gradually reduce temperature (decrease probability of accepting worse solutions)
4. **Stop** when temperature is very low or max iterations reached

### Why It Works for This Problem
- **Large search space**: Thousands of possible outdoor pixel locations
- **Non-convex**: Coverage function likely has many local optima
- **Discrete constraints**: Must place stations only on outdoor pixels
- **Multi-objective aspects**: Need to balance coverage overlap vs. gaps

---

## 2. State Representation

### Current State
A state is a set of base station locations:
```python
state = [
    (x1, y1),  # BS 1 location in meters
    (x2, y2),  # BS 2 location in meters
    ...
    (xN, yN)   # BS N location in meters
]
```

**Constraints**:
- All locations must be outdoor pixels (not buildings)
- Locations must be within map bounds: `[0, MAP_WIDTH_M] x [0, MAP_HEIGHT_M]`

### Alternative Representations
- **Pixel-based**: Store pixel coordinates `(px, py)` instead of meters
  - Pros: Direct indexing, faster neighbor generation
  - Cons: Less precise, discrete steps
- **Index-based**: Store indices into `outdoor_pixels` array
  - Pros: Guaranteed valid, easy to sample
  - Cons: Less intuitive, harder to visualize

**Recommendation**: Use meter coordinates (current approach) for precision, but validate against pixel mask.

---

## 3. Objective Function (Energy/Cost)

### Primary Objective: Maximize Coverage Percentage
```python
def calculate_coverage_percentage(base_stations):
    """
    Calculate what percentage of outdoor pixels are covered.
    
    Returns: coverage_percent (0-100)
    """
    # Run coverage simulation (reuse existing code)
    # Count covered outdoor pixels
    # Return percentage
```

### Energy Function (for SA)
Since SA minimizes, we need to convert:
```python
energy = 100.0 - coverage_percentage  # Minimize this (0 = perfect coverage)
# OR
energy = -coverage_percentage  # Maximize coverage by minimizing negative
```

### Potential Multi-Objective Extensions
1. **Coverage percentage** (primary)
2. **Overlap penalty**: Penalize excessive coverage overlap (wasteful)
3. **Fairness**: Ensure coverage is distributed (not all in one area)
4. **Redundancy**: Reward areas covered by multiple BSs (reliability)

**Weighted combination**:
```python
energy = w1 * (100 - coverage_pct) + w2 * overlap_penalty - w3 * redundancy_bonus
```

---

## 4. Neighbor Generation Strategies

### Strategy 1: Random Walk (Single BS Movement)
- Pick one random base station
- Move it to a nearby outdoor pixel
- **Step size**: Start large (e.g., ±50 meters), shrink with temperature

```python
def generate_neighbor_single_move(state, temperature, outdoor_pixels):
    new_state = state.copy()
    bs_idx = random.randint(0, len(state) - 1)
    
    # Step size decreases with temperature
    max_step = 50.0 * temperature  # meters
    step_x = random.uniform(-max_step, max_step)
    step_y = random.uniform(-max_step, max_step)
    
    new_x = state[bs_idx][0] + step_x
    new_y = state[bs_idx][1] + step_y
    
    # Clamp to valid outdoor location
    new_x, new_y = snap_to_outdoor(new_x, new_y, outdoor_pixels)
    
    new_state[bs_idx] = (new_x, new_y)
    return new_state
```

### Strategy 2: Swap Two Base Stations
- Randomly swap positions of two base stations
- Good for exploring different configurations

### Strategy 3: Multiple Small Moves
- Move all base stations slightly
- Better for fine-tuning near optimum

### Strategy 4: Hybrid Approach
- **High temperature**: Large random moves, swaps
- **Low temperature**: Small local adjustments

### Strategy 5: Smart Neighbor Generation
- **Gradient-based**: Move BS toward uncovered areas
- **Coverage-aware**: Prefer moves that increase coverage
- **Distance-based**: Move BS away from other BSs if too close

**Recommendation**: Start with Strategy 1 (single move with adaptive step size), add Strategy 2 for diversity.

---

## 5. Cooling Schedule

### Temperature Function
```python
T(t) = T0 * alpha^t
# OR
T(t) = T0 / (1 + t)
# OR
T(t) = T0 * exp(-alpha * t)
```

Where:
- `T0`: Initial temperature (start high, e.g., 100.0)
- `t`: Iteration number
- `alpha`: Cooling rate (0.95-0.99 typical)

### Acceptance Probability
```python
def accept_probability(current_energy, neighbor_energy, temperature):
    if neighbor_energy < current_energy:
        return 1.0  # Always accept better solutions
    else:
        delta = neighbor_energy - current_energy
        return math.exp(-delta / temperature)  # Boltzmann distribution
```

### Adaptive Cooling
- **Fast cooling**: If stuck (no improvement for N iterations), cool faster
- **Reheating**: If temperature too low but still improving, slightly increase
- **Temperature reset**: If converged early, restart with higher temperature

---

## 6. Implementation Structure

### Main SA Loop
```python
def simulated_annealing_optimization(
    initial_state,
    objective_func,
    neighbor_func,
    max_iterations=10000,
    initial_temp=100.0,
    cooling_rate=0.95,
    min_temp=0.01
):
    current_state = initial_state
    current_energy = objective_func(current_state)
    best_state = current_state
    best_energy = current_energy
    
    temperature = initial_temp
    
    for iteration in range(max_iterations):
        # Generate neighbor
        neighbor_state = neighbor_func(current_state, temperature)
        neighbor_energy = objective_func(neighbor_state)
        
        # Acceptance decision
        if accept_probability(current_energy, neighbor_energy, temperature) > random.random():
            current_state = neighbor_state
            current_energy = neighbor_energy
        
        # Update best
        if current_energy < best_energy:
            best_state = current_state
            best_energy = current_energy
        
        # Cool down
        temperature *= cooling_rate
        if temperature < min_temp:
            break
        
        # Logging (every N iterations)
        if iteration % 100 == 0:
            print(f"Iter {iteration}: Temp={temperature:.3f}, "
                  f"Energy={current_energy:.2f}, Best={best_energy:.2f}")
    
    return best_state, best_energy
```

### Integration with Existing Code
1. **Extract coverage calculation** into a reusable function:
   ```python
   def calculate_coverage_for_locations(base_stations):
       # Reuse existing coverage calculation logic
       # Return coverage percentage
   ```

2. **Create neighbor generator** that respects outdoor constraint:
   ```python
   def snap_to_outdoor(x, y, outdoor_pixels):
       # Find nearest outdoor pixel to (x, y)
       # Return valid coordinates
   ```

3. **Run SA** before visualization:
   ```python
   if __name__ == "__main__":
       # Run optimization
       optimal_bs = simulated_annealing_optimization(...)
       
       # Visualize with optimal locations
       run_coverage_simulation(optimal_bs)
   ```

---

## 7. Performance Optimizations

### Challenge: Coverage Calculation is Expensive
- Current approach: Check every outdoor pixel for every BS
- For SA: Need to evaluate thousands of states

### Optimization Strategies

1. **Incremental Updates**
   - Only recalculate coverage for moved BS
   - Update coverage map incrementally
   - Track which pixels are covered by which BSs

2. **Early Termination**
   - If neighbor is clearly worse, reject early
   - Use bounds/estimates before full calculation

3. **Caching**
   - Cache coverage for recently evaluated states
   - Use hash of state as key

4. **Sampling**
   - Evaluate coverage on subset of pixels (e.g., every 10th pixel)
   - Full evaluation only for promising candidates

5. **Parallel Evaluation**
   - Evaluate multiple neighbors in parallel
   - Use multiprocessing for independent coverage calculations

6. **Spatial Indexing**
   - Use KD-tree or grid to quickly find nearby outdoor pixels
   - Accelerate neighbor generation

### Recommended Approach
- Start with **full evaluation** to ensure correctness
- Add **incremental updates** for 10-100x speedup
- Use **sampling** only if still too slow

---

## 8. Hyperparameters to Tune

### Critical Parameters
1. **Initial Temperature (`T0`)**
   - Too high: Accepts too many bad moves (wastes time)
   - Too low: Gets stuck in local optima
   - **Tuning**: Start with `T0 = 10.0 * initial_energy`

2. **Cooling Rate (`alpha`)**
   - Too fast: Converges to local optimum
   - Too slow: Takes forever
   - **Typical**: 0.95-0.99

3. **Max Iterations**
   - Depends on problem size and time budget
   - **Start**: 5000-10000 iterations

4. **Step Size**
   - Initial: ~10% of map dimension (e.g., 50 meters)
   - Should decrease with temperature

5. **Minimum Temperature**
   - Stop when `T < 0.01` or similar

### Adaptive Tuning
- Monitor acceptance rate (should be ~40-60% early, ~0% late)
- Adjust cooling rate if acceptance rate drops too fast/slow

---

## 9. Validation & Testing

### Baseline Comparison
- Compare SA result vs. random placement
- Compare SA result vs. grid-based placement
- Compare SA result vs. greedy placement

### Convergence Analysis
- Plot energy vs. iteration
- Plot temperature vs. iteration
- Plot acceptance rate vs. iteration

### Reproducibility
- Use fixed random seed
- Run multiple times with different seeds
- Report mean/std of final coverage

### Visualization
- Show optimization progress (animation)
- Compare initial vs. final placement
- Show coverage improvement

---

## 10. Potential Challenges & Solutions

### Challenge 1: Stuck in Local Optima
**Symptoms**: Coverage plateaus, no improvement
**Solutions**:
- Increase initial temperature
- Add occasional large random jumps
- Use multiple restarts
- Try different neighbor generation strategies

### Challenge 2: Too Slow
**Symptoms**: Takes hours to run
**Solutions**:
- Implement incremental coverage updates
- Use sampling for evaluation
- Reduce max iterations
- Parallelize neighbor evaluation

### Challenge 3: Invalid States
**Symptoms**: Base stations placed on buildings
**Solutions**:
- Always validate and snap to outdoor pixels
- Use index-based representation
- Add constraint checking in neighbor generation

### Challenge 4: Poor Coverage Despite Optimization
**Symptoms**: SA finds solution but coverage still low
**Solutions**:
- Check if problem is feasible (enough BSs for area?)
- Adjust SNR threshold or transmit power
- Consider multi-objective optimization (redundancy, fairness)

---

## 11. Extension Ideas

### Multi-Objective Optimization
- Pareto-optimal solutions (coverage vs. cost)
- Weighted sum or NSGA-II algorithm

### Constrained Optimization
- Minimum distance between BSs
- Maximum distance from certain areas
- Budget constraints (different BS types with different costs)

### Dynamic Optimization
- Time-varying coverage requirements
- Moving obstacles or changing environment

### Hybrid Approaches
- SA + Local Search: Use SA to find good region, then local search
- SA + Genetic Algorithm: Combine with GA for diversity
- Multi-start SA: Run multiple SA runs, take best

---

## 12. Implementation Checklist

- [ ] Extract `calculate_coverage_percentage()` function
- [ ] Implement `snap_to_outdoor()` helper
- [ ] Create `generate_neighbor()` function
- [ ] Implement cooling schedule
- [ ] Create main SA loop
- [ ] Add logging/progress tracking
- [ ] Integrate with existing visualization
- [ ] Add parameter configuration
- [ ] Test with small number of BSs first
- [ ] Optimize coverage calculation (incremental updates)
- [ ] Create comparison plots (before/after)
- [ ] Document hyperparameters

---

## 13. Expected Results

### Performance Metrics
- **Coverage improvement**: Expect 10-30% increase over random
- **Runtime**: 1-10 minutes depending on optimizations
- **Convergence**: Should see steady improvement, then plateau

### Success Criteria
- Coverage > 80% (if feasible)
- Consistent results across multiple runs
- Visually reasonable placement (not clustered)

---

## 14. Next Steps

1. **Phase 1**: Basic SA implementation
   - Simple neighbor generation
   - Full coverage evaluation
   - Basic cooling schedule

2. **Phase 2**: Optimization
   - Incremental coverage updates
   - Adaptive cooling
   - Better neighbor generation

3. **Phase 3**: Analysis
   - Compare with baselines
   - Hyperparameter tuning
   - Visualization improvements

---

## References & Resources

- **Simulated Annealing Algorithm**: Classic optimization metaheuristic
- **Coverage Optimization**: Common in wireless network planning
- **Python Libraries**: Consider `scipy.optimize` for advanced SA variants
- **Related Algorithms**: Genetic Algorithms, Particle Swarm Optimization, Tabu Search

---

*This document serves as a brainstorming guide. Implementation details may evolve based on testing and results.*
