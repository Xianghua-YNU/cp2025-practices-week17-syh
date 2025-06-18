#!/usr/bin/env python3
"""
Module: Finite Thickness Parallel Plate Capacitor (Student Version)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace

def solve_laplace_sor(nx, ny, plate_thickness, plate_separation, omega=1.9, max_iter=10000, tolerance=1e-4):
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
        np.ndarray: 2D electric potential distribution
    """
    # 参数验证
    if nx <= 0 or ny <= 0:
        raise ValueError("Grid dimensions (nx, ny) must be positive integers")
    if plate_thickness <= 0 or plate_separation <= 0:
        raise ValueError("Plate thickness and separation must be positive integers")
    if not (1.0 < omega < 2.0):
        raise ValueError("Relaxation factor omega must be in range (1.0, 2.0)")
    
    # 初始化电势网格
    potential = np.zeros((ny, nx))
    
    # 定义导体区域（极板）
    conductor_left = nx // 4
    conductor_right = 3 * nx // 4
    
    # 上极板 (+100V)
    upper_plate_top = ny // 2 + plate_separation // 2
    upper_plate_bottom = upper_plate_top + plate_thickness
    potential[upper_plate_top:upper_plate_bottom, conductor_left:conductor_right] = 100.0
    
    # 下极板 (-100V)
    lower_plate_bottom = ny // 2 - plate_separation // 2
    lower_plate_top = lower_plate_bottom - plate_thickness
    potential[lower_plate_top:lower_plate_bottom, conductor_left:conductor_right] = -100.0
    
    # 创建导体掩码（用于保持导体区域电势恒定）
    conductor_mask = np.zeros_like(potential, dtype=bool)
    conductor_mask[upper_plate_top:upper_plate_bottom, conductor_left:conductor_right] = True
    conductor_mask[lower_plate_top:lower_plate_bottom, conductor_left:conductor_right] = True
    
    # 设置边界条件（接地）
    potential[:, 0] = 0.0      # 左边界
    potential[:, -1] = 0.0     # 右边界
    potential[0, :] = 0.0      # 上边界
    potential[-1, :] = 0.0     # 下边界
    
    # SOR迭代求解
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    convergence_history = []
    
    for iteration in range(max_iter):
        max_diff = 0.0
        
        # 逐点更新（Gauss-Seidel风格）
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                if conductor_mask[i, j]:
                    continue
                    
                old_value = potential[i, j]
                new_value = 0.25 * (
                    potential[i+1, j] + potential[i-1, j] + 
                    potential[i, j+1] + potential[i, j-1]
                )
                potential[i, j] = (1 - omega) * old_value + omega * new_value
                
                diff = abs(new_value - old_value)
                if diff > max_diff:
                    max_diff = diff
        
        convergence_history.append(max_diff)
        
        # 检查收敛
        if max_diff < tolerance:
            print(f"Converged after {iteration+1} iterations with max_diff = {max_diff}")
            break
            
    else:
        print(f"Warning: Did not converge within {max_iter} iterations. Max_diff = {max_diff}")
    
    return potential, convergence_history, iteration+1

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
    # 使用scipy的laplace函数计算拉普拉斯算子
    laplacian = laplace(potential_grid, mode='nearest')
    
    # 考虑网格间距的影响
    charge_density = -laplacian / (4 * np.pi * dx * dy)
    
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
    
    # 绘制电势分布
    im1 = ax1.imshow(potential, cmap='viridis', origin='lower',
                    extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]])
    ax1.set_title('Electric Potential Distribution')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    fig.colorbar(im1, ax=ax1, label='Potential (V)')
    
    # 绘制电荷密度分布
    max_charge = max(abs(np.min(charge_density)), abs(np.max(charge_density)))
    im2 = ax2.imshow(charge_density, cmap='coolwarm', origin='lower',
                    extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]],
                    vmin=-max_charge, vmax=max_charge)
    ax2.set_title('Charge Density Distribution')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    fig.colorbar(im2, ax=ax2, label='Charge Density')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 模拟参数
    nx = 100
    ny = 100
    plate_thickness = 4
    plate_separation = 20
    
    # 创建坐标数组
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    # 求解拉普拉斯方程
    potential, _, _ = solve_laplace_sor(nx, ny, plate_thickness, plate_separation)
    
    # 计算电荷密度
    charge_density = calculate_charge_density(potential, dx, dy)
    
    # 可视化结果
    plot_results(potential, charge_density, x, y)    
