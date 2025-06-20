import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

def solve_laplace_jacobi(xgrid, ygrid, w, d, tol=1e-5):
    """
    使用Jacobi迭代法求解拉普拉斯方程
    
    参数:
        xgrid (int): x方向网格点数
        ygrid (int): y方向网格点数
        w (int): 平行板宽度
        d (int): 平行板间距
        tol (float): 收敛容差
    
    返回:
        tuple: (potential_array, iterations, convergence_history)
    
    物理背景: 求解平行板电容器内部的电势分布，满足拉普拉斯方程 \(\nabla^2 V = 0\)。
    数值方法: 使用Jacobi迭代法，通过反复迭代更新每个网格点的电势值，直至收敛。
    
    实现步骤:
    1. 初始化电势网格，设置边界条件（极板电势）。
    2. 循环迭代，每次迭代根据周围点的电势更新当前点的电势。
    3. 记录每次迭代的最大变化量，用于收敛历史分析。
    4. 检查收敛条件，如果最大变化量小于容差，则停止迭代。
    5. 返回最终的电势分布、迭代次数和收敛历史。
    """
    # 初始化电势网格
    u = np.zeros((ygrid, xgrid))
    
    # 计算平行板位置
    xL = (xgrid - w) // 2
    xR = (xgrid + w) // 2
    yB = (ygrid - d) // 2
    yT = (ygrid + d) // 2
    
    # 设置平行板边界条件
    u[yB, xL:xR+1] = -100  # 下板
    u[yT, xL:xR+1] = 100   # 上板
    
    iterations = 0
    convergence_history = []
    max_diff = tol + 1  # 初始化最大变化量，确保进入循环
    
    while max_diff > tol:
        u_old = u.copy()
        max_diff = 0
        
        # 内部点迭代更新（不包括边界）
        for i in range(1, ygrid-1):
            for j in range(1, xgrid-1):
                # 跳过平行板区域
                if (i == yB or i == yT) and (xL <= j <= xR):
                    continue
                    
                # Jacobi迭代更新
                u[i, j] = 0.25 * (u_old[i+1, j] + u_old[i-1, j] + u_old[i, j+1] + u_old[i, j-1])
                
                # 计算最大变化量
                diff = abs(u[i, j] - u_old[i, j])
                if diff > max_diff:
                    max_diff = diff
        
        iterations += 1
        convergence_history.append(max_diff)
    
    return u, iterations, convergence_history

def solve_laplace_sor(xgrid, ygrid, w, d, omega=1.25, Niter=1000, tol=1e-5):
    """
    实现SOR算法求解平行板电容器的电势分布
    
    参数:
        xgrid (int): x方向网格点数
        ygrid (int): y方向网格点数
        w (int): 平行板宽度
        d (int): 平行板间距
        omega (float): 松弛因子
        Niter (int): 最大迭代次数
        tol (float): 收敛容差
    返回:
        tuple: (电势分布数组, 迭代次数, 收敛历史)
    
    物理背景: 求解平行板电容器内部的电势分布，满足拉普拉斯方程 \(\nabla^2 V = 0\)。
    数值方法: 使用逐次超松弛（SOR）迭代法，通过引入松弛因子加速收敛。
    
    实现步骤:
    1. 初始化电势网格，设置边界条件（极板电势）。
    2. 循环迭代，每次迭代根据周围点和松弛因子更新当前点的电势。
    3. 记录每次迭代的最大变化量，用于收敛历史分析。
    4. 检查收敛条件，如果最大变化量小于容差或达到最大迭代次数，则停止迭代。
    5. 返回最终的电势分布、迭代次数和收敛历史。
    """
    # 初始化电势网格
    u = np.zeros((ygrid, xgrid))
    
    # 计算平行板位置
    xL = (xgrid - w) // 2
    xR = (xgrid + w) // 2
    yB = (ygrid - d) // 2
    yT = (ygrid + d) // 2
    
    # 设置平行板边界条件
    u[yB, xL:xR+1] = -100  # 下板
    u[yT, xL:xR+1] = 100   # 上板
    
    iterations = 0
    convergence_history = []
    max_diff = tol + 1  # 初始化最大变化量，确保进入循环
    
    # 修正：添加最大迭代次数检查
    while iterations < Niter and max_diff > tol:
        max_diff = 0
        
        # 内部点迭代更新（不包括边界）
        for i in range(1, ygrid-1):
            for j in range(1, xgrid-1):
                # 跳过平行板区域
                if (i == yB or i == yT) and (xL <= j <= xR):
                    continue
                
                # 计算SOR迭代的中间值
                r_ij = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1])
                
                # SOR迭代更新
                old_value = u[i, j]
                u[i, j] = (1 - omega) * old_value + omega * r_ij
                
                # 计算最大变化量
                diff = abs(u[i, j] - old_value)
                if diff > max_diff:
                    max_diff = diff
        
        iterations += 1
        convergence_history.append(max_diff)
    
    return u, iterations, convergence_history

def plot_results(x, y, u, method_name):
    """
    绘制三维电势分布、等势线和电场线
    
    参数:
        x (array): X坐标数组
        y (array): Y坐标数组
        u (array): 电势分布数组
        method_name (str): 方法名称
    
    实现步骤:
    1. 创建包含两个子图的图形。
    2. 在第一个子图中绘制三维线框图显示电势分布以及在z方向的投影等势线。
    3. 在第二个子图中绘制等势线和电场线流线图。
    4. 设置图表标题、标签和显示(注意不要出现乱码)。
    """
    # 创建图形和子图
    fig = plt.figure(figsize=(14, 6))
    
    # 第一个子图：三维电势分布和等势线投影
    ax1 = fig.add_subplot(121, projection='3d')
    
    # 绘制三维线框图
    X, Y = np.meshgrid(x, y)
    surf = ax1.plot_wireframe(X, Y, u, rstride=2, cstride=2, color='blue', linewidth=0.5)
    
    # 在z方向上投影等势线
    cset = ax1.contourf(X, Y, u, zdir='z', offset=np.min(u), cmap='viridis', alpha=0.6)
    
    # 设置坐标轴标签和标题
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('电势 (V)')
    ax1.set_title(f'{method_name} - 三维电势分布')
    
    # 第二个子图：等势线和电场线
    ax2 = fig.add_subplot(122)
    
    # 计算电场分量（负电势梯度）
    EY, EX = np.gradient(-u)
    
    # 绘制等势线
    contour = ax2.contour(X, Y, u, 20, colors='black', linewidths=0.8)
    ax2.clabel(contour, inline=True, fontsize=8)
    
    # 绘制电场线
    ax2.streamplot(X, Y, EX, EY, density=1.5, color='red', linewidth=1, arrowsize=1.5)
    
    # 设置坐标轴标签和标题
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title(f'{method_name} - 等势线（黑线）和电场线（红线）')
    
    # 调整布局
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 模拟参数设置
    xgrid = 100  # x方向网格点数
    ygrid = 100  # y方向网格点数
    w = 60       # 平行板宽度
    d = 20       # 平行板间距
    omega = 1.8  # SOR松弛因子
    
    # 创建网格坐标
    x = np.linspace(0, 1, xgrid)
    y = np.linspace(0, 1, ygrid)
    
    # 使用Jacobi方法求解
    start_time = time.time()
    u_jacobi, iterations_jacobi, history_jacobi = solve_laplace_jacobi(xgrid, ygrid, w, d)
    jacobi_time = time.time() - start_time
    print(f"Jacobi方法迭代次数: {iterations_jacobi}")
    print(f"Jacobi方法计算时间: {jacobi_time:.4f}秒")
    
    # 使用SOR方法求解
    start_time = time.time()
    u_sor, iterations_sor, history_sor = solve_laplace_sor(xgrid, ygrid, w, d, omega)
    sor_time = time.time() - start_time
    print(f"SOR方法迭代次数: {iterations_sor}")
    print(f"SOR方法计算时间: {sor_time:.4f}秒")
    
    # 绘制结果
    plot_results(x, y, u_jacobi, "Jacobi方法")
    plot_results(x, y, u_sor, f"SOR方法 (ω={omega})")
    
    # 比较收敛速度
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, iterations_jacobi+1), history_jacobi, label='Jacobi')
    plt.semilogy(range(1, iterations_sor+1), history_sor, label=f'SOR (ω={omega})')
    plt.xlabel('迭代次数')
    plt.ylabel('收敛误差 (对数刻度)')
    plt.title('收敛速度比较')
    plt.legend()
    plt.grid(True)
    plt.show()    
