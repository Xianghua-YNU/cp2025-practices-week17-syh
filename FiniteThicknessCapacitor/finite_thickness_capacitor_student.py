#!/usr/bin/env python3
"""
Module: Finite Thickness Parallel Plate Capacitor (Student Version)
"""

import numpy as np
import matplotlib.pyplot as plt

def solve_laplace_sor(nx, ny, plate_thickness, plate_separation, omega=1.9, max_iter=10000, tolerance=1e-6):
    """
    Solve 2D Laplace equation using Successive Over-Relaxation (SOR) method
    for finite thickness parallel plate capacitor.
    
    Args:
        nx (int): Number of grid points in x direction
        ny (int): Number of grid points in y direction  
        plate_thickness (int): Thickness of conductor plates in grid points
        plate_separation (int): Separation between plates in grid points
        omega (float): Relaxation factor (1.0 < omega < 2.0)
        max_iter (int): Maximum number of iterations
        tolerance (float): Convergence tolerance
        
    Returns:
        tuple: (potential_grid, convergence_history, conductor_mask)
            - potential_grid: 2D array of electric potential
            - convergence_history: List of maximum errors during iterations
            - conductor_mask: Boolean array marking conductor regions
    """
    # Initialize potential grid
    potential_grid = np.zeros((ny, nx))
    
    # Create conductor mask
    conductor_mask = np.zeros((ny, nx), dtype=bool)
    
    # Define conductor regions
    conductor_left = nx // 4
    conductor_right = 3 * nx // 4
    y_upper_start = ny // 2 + plate_separation // 2
    y_upper_end = y_upper_start + plate_thickness
    conductor_mask[y_upper_start:y_upper_end, conductor_left:conductor_right] = True
    potential_grid[y_upper_start:y_upper_end, conductor_left:conductor_right] = 100.0
    
    y_lower_end = ny // 2 - plate_separation // 2
    y_lower_start = y_lower_end - plate_thickness
    conductor_mask[y_lower_start:y_lower_end, conductor_left:conductor_right] = True
    potential_grid[y_lower_start:y_lower_end, conductor_left:conductor_right] = -100.0
    
    # Boundary conditions: grounded sides
    potential_grid[:, 0] = 0.0
    potential_grid[:, -1] = 0.0
    potential_grid[0, :] = 0.0
    potential_grid[-1, :] = 0.0
    
    # SOR iteration
    convergence_history = []
    for iteration in range(max_iter):
        max_error = 0.0
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                if not conductor_mask[i, j]:  # Skip conductor points
                    new_value = 0.25 * (
                        potential_grid[i + 1, j] + potential_grid[i - 1, j] +
                        potential_grid[i, j + 1] + potential_grid[i, j - 1]
                    )
                    new_value = (1 - omega) * potential_grid[i, j] + omega * new_value
                    error = abs(new_value - potential_grid[i, j])
                    potential_grid[i, j] = new_value
                    max_error = max(max_error, error)
        
        convergence_history.append(max_error)
        if max_error < tolerance:
            print(f"Converged after {iteration + 1} iterations with max error: {max_error}")
            break
    else:
        print(f"Warning: Did not converge within {max_iter} iterations. Final error: {max_error}")
    
    return potential_grid, convergence_history, conductor_mask

def calculate_charge_density(potential_grid, dx, dy):
    """
    Calculate charge density using Poisson equation: rho = -1/(4*pi) * nabla^2(U)
    
    Args:
        potential_grid (np.ndarray): 2D electric potential distribution
        dx (float): Grid spacing in x direction
        dy (float): Grid spacing in y direction
        
    Returns:
        np.ndarray: 2D charge density distribution
    """
    # Calculate Laplacian using finite difference method
    laplacian = np.zeros_like(potential_grid)
    laplacian[1:-1, 1:-1] = (
        (potential_grid[:-2, 1:-1] + potential_grid[2:, 1:-1] +
         potential_grid[1:-1, :-2] + potential_grid[1:-1, 2:] -
         4 * potential_grid[1:-1, 1:-1]) / (dx * dy)
    )
    
    # Charge density from Poisson equation
    charge_density = -laplacian / (4 * np.pi)
    return charge_density

def plot_results(potential, charge_density, x_coords, y_coords):
    """
    Create visualization of potential and charge density distributions.
    
    Args:
        potential (np.ndarray): 2D electric potential distribution
        charge_density (np.ndarray): Charge density distribution
        x_coords (np.ndarray): X coordinate array
        y_coords (np.ndarray): Y coordinate array
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot potential distribution
    contour = ax1.contourf(x_coords, y_coords, potential, levels=50, cmap='viridis')
    ax1.set_title('Electric Potential Distribution')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    fig.colorbar(contour, ax=ax1, label='Potential (V)')
    
    # Plot charge density distribution
    im = ax2.imshow(charge_density, cmap='seismic', origin='lower', 
                   extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]])
    ax2.set_title('Charge Density Distribution')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    fig.colorbar(im, ax=ax2, label='Charge Density')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Simulation parameters
    nx, ny = 120, 100
    plate_thickness = 10
    plate_separation = 40
    omega = 1.9
    
    # Physical dimensions
    Lx, Ly = 1.0, 1.0
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    
    # Create coordinate arrays
    x_coords = np.linspace(0, Lx, nx)
    y_coords = np.linspace(0, Ly, ny)
    
    print("Solving finite thickness parallel plate capacitor...")
    print(f"Grid size: {nx} x {ny}")
    print(f"Plate thickness: {plate_thickness} grid points")
    print(f"Plate separation: {plate_separation} grid points")
    print(f"SOR relaxation factor: {omega}")
    
    # Solve Laplace equation
    potential, convergence_history, conductor_mask = solve_laplace_sor(nx, ny, plate_thickness, plate_separation, omega)
    
    # Calculate charge density
    charge_density = calculate_charge_density(potential, dx, dy)
    
    # Visualize results
    plot_results(potential, charge_density, x_coords, y_coords)
