#!/usr/bin/env python3
"""
Module: Finite Thickness Parallel Plate Capacitor (Student Version)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace

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
        list: Convergence history
        np.ndarray: Conductor mask
    """
    # 参数验证
    if nx <= 0 or ny <= 0:
        raise ValueError("Grid dimensions must be positive integers")
    if plate_thickness <= 0 or plate_separation <= 0:
        raise ValueError("Plate thickness and separation must be positive integers")
    if not (1.0 omega < < 2.0):
        raise ValueError("Relaxation factor omega must be in (1.0, 2.0)")
    
    # 初始化电势网格
    potential = np.zeros((ny, nx))
    
    # 创建导体掩码
    conductor_mask = np.zeros((ny, nx), dtype=bool)
    
    # 设置导体区域（极板）
    conductor_left = nx // 4
    conductor_right = 3 * nx // 4
    
    # 上极板 (+100V)
    upper_plate_top = ny // 2 + plate_separation // 2
    upper_plate_bottom = upper_plate_top + plate_thickness
    conductor_mask[upper_plate_top:upper_plate_bottom, conductor_left:conductor_right] = True
    potential[upper_plate_top:upper_plate_bottom, conductor_left:conductor_right] = 100.0
    
    # 下极板 (-100V)
    lower_plate_bottom = ny // 2 - plate_separation // 2
    lower_plate_top = lower_plate_bottom - plate_thickness
    conductor_mask[lower_plate_top:lower_plate_bottom, conductor_left:conductor_right] = True
    potential[lower_plate_top:lower_plate_bottom, conductor_left:conductor_right] = -100.0
    
    # 设置边界条件（接地）
    potential[:, 0] = 0.0      # 左边界
    potential[:, -1] = 0.0     # 右边界
    potential[0, :] = 0.0      # 上边界
    potential[-1, :] = 0.0     # 下边界
    
    # SOR迭代求解
    convergence_history = []
    converged = False
    
    for iteration in range(max_iter):
        max_diff = 0.0
        old_potential = potential.copy()
        
        # 逐点更新（Gauss-Seidel风格）
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                if conductor_mask[i, j]:
                    continue
                    
                # SOR更新公式
                neighbor_sum = (
                    potential[i+1, j] + potential[i-1, j] + 
                    potential[i, j+1] + potential[i, j-1]
                )
                new_value = (1 - omega) * potential[i, j] + omega * 0.25 * neighbor_sum
                potential[i, j] = new_value
                
                # 跟踪最大差异
                diff = abs(new_value - old_potential[i, j])
                if diff > max_diff:
                    max_diff = diff
        
        convergence_history.append(max_diff)
        
        # 检查收敛
        if max_diff < tolerance:
            converged = True
            print(f"Converged after {iteration+1} iterations with max_diff = {max_diff}")
            break
            
    else:
        print(f"Warning: Did not converge within {max_iter} iterations. Max_diff = {max_diff}")
    
    # 返回与参考答案兼容的格式
    return potential, convergence_history, conductor_mask

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
    # 使用离散拉普拉斯算子显式计算
    laplacian = (potential_grid[:-2, 1:-1] + potential_grid[2:, 1:-1] + 
                 potential_grid[1:-1, :-2] + potential_grid[1:-1, 2:] - 
                 4 * potential_grid[1:-1, 1:-1]) / (dx * dy)
    
    # 电荷密度计算，注意网格间距的处理
    charge_density = np.zeros_like(potential_grid)
    charge_density[1:-1, 1:-1] = -laplacian / (4 * np.pi)
    
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
    
    # 绘制电势分布（等高线图）
    contour = ax1.contourf(x_coords, y_coords, potential, levels=50, cmap='viridis')
    ax1.set_title('Electric Potential Distribution')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    fig.colorbar(contour, ax=ax1, label='Potential (V)')
    
    # 绘制电荷密度分布
    vmax = max(abs(np.min(charge_density)), abs(np.max(charge_density)))
    im = ax2.imshow(charge_density, cmap='seismic', origin='lower',
                   extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]],
                   vmin=-vmax, vmax=vmax)
    ax2.set_title('Charge Density Distribution')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    fig.colorbar(im, ax=ax2, label='Charge Density')
    
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
    potential, convergence_history, conductor_mask = solve_laplace_sor(nx, ny, plate_thickness, plate_separation)
    
    # 计算电荷密度
    charge_density = calculate_charge_density(potential, dx, dy)
    
    # 可视化结果
    plot_results(potential, charge_density, x, y)
