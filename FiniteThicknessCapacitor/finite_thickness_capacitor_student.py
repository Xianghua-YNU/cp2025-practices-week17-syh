#!/usr/bin/env python3
"""
Module: Finite Thickness Parallel Plate Capacitor (Student Version)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
from matplotlib.colors import TwoSlopeNorm

def solve_laplace_sor(nx, ny, plate_thickness, plate_separation, omega=1.9, max_iter=10000, tolerance=1e-4):
    """
    Solve 2D Laplace equation using SOR method for finite thickness parallel plate capacitor.
    
    Args:
        nx (int): Number of grid points in x direction
        ny (int): Number of grid points in y direction
        plate_thickness (int): Thickness of conductor plates in grid points
        plate_separation (int): Separation between plates in grid points
        omega (float): Relaxation factor (1.0 < omega < 2.0)
        max_iter (int): Maximum number of iterations
        tolerance (float): Convergence tolerance
        
    Returns:
        np.ndarray: 2D electric potential distribution
    """
    # 参数验证
    if nx <= 0 or ny <= 0:
        raise ValueError("Grid dimensions (nx, ny) must be positive integers")
    if plate_thickness <= 0 or plate_separation <= 0:
        raise ValueError("Plate thickness and separation must be positive integers")
    if not (1.0 < omega < 2.0):
        raise ValueError("Relaxation factor omega must be in range (1.0, 2.0)")
        
    # Initialize potential grid with zeros
    potential = np.zeros((ny, nx))
    
    # Calculate plate positions
    plate_y_center = ny // 2
    upper_plate_top = plate_y_center - plate_separation // 2
    upper_plate_bottom = upper_plate_top - plate_thickness + 1
    lower_plate_top = plate_y_center + plate_separation // 2
    lower_plate_bottom = lower_plate_top + plate_thickness - 1
    
    # 设置所有边界条件
    potential[:, 0] = 0.0   # 左边界
    potential[:, -1] = 0.0  # 右边界
    potential[0, :] = 0.0   # 上边界
    potential[-1, :] = 0.0  # 下边界
    
    # Initialize plates with fixed potentials
    potential[upper_plate_bottom:upper_plate_top+1, :] = 100.0
    potential[lower_plate_top:lower_plate_bottom+1, :] = -100.0
    
    # Define mask for conductor regions to maintain fixed potential
    conductor_mask = np.zeros_like(potential, dtype=bool)
    conductor_mask[upper_plate_bottom:upper_plate_top+1, :] = True
    conductor_mask[lower_plate_top:lower_plate_bottom+1, :] = True
    
    # SOR iteration
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    coeff = omega / (2.0 * (1.0 + (dx/dy)**2))
    
    for iteration in range(max_iter):
        max_diff = 0.0
        old_potential = potential.copy()
        
        # Update interior points using SOR
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                # Skip conductor regions
                if conductor_mask[j, i]:
                    continue
                    
                # SOR update
                neighbor_avg = (old_potential[j, i+1] + old_potential[j, i-1]) * (dy**2)
                neighbor_avg += (old_potential[j+1, i] + old_potential[j-1, i]) * (dx**2)
                neighbor_avg /= 2.0 * (dx**2 + dy**2)
                
                new_value = (1.0 - omega) * old_potential[j, i] + omega * neighbor_avg
                potential[j, i] = new_value
                
                # Track maximum difference for convergence check
                diff = abs(new_value - old_potential[j, i])
                if diff > max_diff:
                    max_diff = diff
        
        # Check convergence
        if max_diff < tolerance:
            print(f"Converged after {iteration+1} iterations with max_diff = {max_diff}")
            break
            
    else:
        print(f"Warning: Did not converge within {max_iter} iterations. Max_diff = {max_diff}")
    
    return potential

def calculate_charge_density(potential_grid, dx, dy):
    """
    Calculate charge density using Poisson equation.
    
    Args:
        potential_grid (np.ndarray): 2D electric potential distribution
        dx (float): Grid spacing in x direction
        dy (float): Grid spacing in y direction
        
    Returns:
        np.ndarray: 2D charge density distribution
    """
    # Use scipy's laplace function with appropriate spacing
    # Scale factor accounts for grid spacing in both directions
    scale_factor = 1.0 / (4.0 * np.pi)
    
    # Compute Laplacian using central differences with proper scaling
    laplacian = laplace(potential_grid, mode='nearest') / (dx * dy)
    
    # Charge density from Poisson's equation: ρ = -∇²U / (4π)
    charge_density = -scale_factor * laplacian
    
    return charge_density

def plot_results(potential, charge_density, x_coords, y_coords, plate_thickness, plate_separation):
    """
    Create visualization of potential and charge density distributions.
    
    Args:
        potential (np.ndarray): 2D electric potential distribution
        charge_density (np.ndarray): Charge density distribution
        x_coords (np.ndarray): X coordinate array
        y_coords (np.ndarray): Y coordinate array
        plate_thickness (int): Thickness of conductor plates in grid points
        plate_separation (int): Separation between plates in grid points
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Calculate plate positions for visualization
    plate_y_center = len(y_coords) // 2
    upper_plate_top = plate_y_center - plate_separation // 2
    upper_plate_bottom = upper_plate_top - plate_thickness + 1
    lower_plate_top = plate_y_center + plate_separation // 2
    lower_plate_bottom = lower_plate_top + plate_thickness - 1
    
    # Plot potential distribution
    im1 = ax1.imshow(potential, cmap='viridis', origin='lower', 
                    extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]])
    ax1.set_title('Electric Potential Distribution')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    fig.colorbar(im1, ax=ax1, label='Potential (V)')
    
    # Plot equipotential lines
    contour_levels = np.linspace(-100, 100, 11)
    cs = ax1.contour(x_coords, y_coords, potential, levels=contour_levels, colors='white', linestyles='-', alpha=0.5)
    ax1.clabel(cs, inline=True, fontsize=8)
    
    # Highlight conductor plates
    ax1.axhspan(y_coords[upper_plate_bottom], y_coords[upper_plate_top], color='red', alpha=0.3)
    ax1.axhspan(y_coords[lower_plate_top], y_coords[lower_plate_bottom], color='blue', alpha=0.3)
    
    # Plot charge density distribution
    # Center colormap at zero for better visualization of positive and negative charges
    norm = TwoSlopeNorm(vmin=np.min(charge_density), vcenter=0, vmax=np.max(charge_density))
    im2 = ax2.imshow(charge_density, cmap='coolwarm', norm=norm, origin='lower',
                    extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]])
    ax2.set_title('Charge Density Distribution')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    fig.colorbar(im2, ax=ax2, label='Charge Density')
    
    # Highlight conductor plates
    ax2.axhspan(y_coords[upper_plate_bottom], y_coords[upper_plate_top], color='gray', alpha=0.3)
    ax2.axhspan(y_coords[lower_plate_top], y_coords[lower_plate_bottom], color='gray', alpha=0.3)
    
    # Plot charge density along plate surfaces
    # Extract charge density along the surfaces of the plates
    upper_surface_charge = charge_density[upper_plate_top, :]
    lower_surface_charge = charge_density[lower_plate_top, :]
    
    ax3.plot(x_coords, upper_surface_charge, 'r-', label='Upper Plate Surface')
    ax3.plot(x_coords, lower_surface_charge, 'b-', label='Lower Plate Surface')
    ax3.set_title('Surface Charge Density')
    ax3.set_xlabel('x')
    ax3.set_ylabel('Charge Density')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Simulation parameters
    nx = 100  # Grid points in x-direction
    ny = 100  # Grid points in y-direction
    plate_thickness = 4  # Thickness of plates in grid points (at least 2Δ)
    plate_separation = 20  # Separation between plates in grid points
    
    # Create coordinate arrays
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    # Solve Laplace equation
    potential = solve_laplace_sor(nx, ny, plate_thickness, plate_separation)
    
    # Calculate charge density
    charge_density = calculate_charge_density(potential, dx, dy)
    
    # Plot results
    plot_results(potential, charge_density, x, y, plate_thickness, plate_separation)    
