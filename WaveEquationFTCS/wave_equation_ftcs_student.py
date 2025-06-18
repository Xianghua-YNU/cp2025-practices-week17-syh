import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def u_t(x, C=1, d=0.1, sigma=0.3, L=1):
    """
    计算初始速度剖面 psi(x)。

    参数:
        x (np.ndarray): 位置数组。
        C (float): 振幅常数。
        d (float): 指数项的偏移量。
        sigma (float): 指数项的宽度。
        L (float): 弦的长度。
    返回:
        np.ndarray: 初始速度剖面。
    """
    return C * x * (L - x) / L**2 * np.exp(-(x - d)**2 / (2 * sigma**2))

def solve_wave_equation_ftcs(parameters):
    """
    使用FTCS有限差分法求解一维波动方程。
    
    参数:
        parameters (dict): 包含以下参数的字典：
            - 'a': 波速 (m/s)。
            - 'L': 弦的长度 (m)。
            - 'd': 初始速度剖面的偏移量 (m)。
            - 'C': 初始速度剖面的振幅常数 (m/s)。
            - 'sigma': 初始速度剖面的宽度 (m)。
            - 'dx': 空间步长 (m)。
            - 'dt': 时间步长 (s)。
            - 'total_time': 总模拟时间 (s)。
    返回:
        tuple: 包含以下内容的元组：
            - np.ndarray: 解数组 u(x, t)。
            - np.ndarray: 空间数组 x。
            - np.ndarray: 时间数组 t。
    """
    # 从 parameters 字典中获取所有必要的物理和数值参数
    a = parameters['a']
    L = parameters['L']
    d = parameters['d']
    C = parameters['C']
    sigma = parameters['sigma']
    dx = parameters['dx']
    dt = parameters['dt']
    total_time = parameters['total_time']

    # 初始化空间网格 x 和时间网格 t
    nx = int(L / dx) + 1  # 空间点数
    nt = int(total_time / dt) + 1  # 时间点数
    x = np.linspace(0, L, nx)
    t = np.linspace(0, total_time, nt)

    # 创建一个零数组 u 来存储解，其维度为 (x.size, t.size)
    u = np.zeros((nx, nt))

    # 计算稳定性条件 c = (a * dt / dx)^2
    c = (a * dt / dx)**2
    if c >= 1:
        print(f"警告: 稳定性条件不满足，c = {c:.4f} >= 1")

    # 应用初始条件：u(x, 0) = 0
    # 由于 u 数组已经初始化为零，因此不需要额外操作

    # 计算第一个时间步 u(x, 1) 的值
    psi = u_t(x, C, d, sigma, L)
    u[:, 1] = psi * dt  # 简化形式，因为 u(x, 0) = 0

    # 使用FTCS方案迭代计算后续时间步的解
    for j in range(1, nt - 1):
        # 内部点更新
        for i in range(1, nx - 1):
            u[i, j + 1] = c * (u[i + 1, j] + u[i - 1, j]) + 2 * (1 - c) * u[i, j] - u[i, j - 1]
        
        # 边界条件处理
        u[0, j + 1] = 0  # 左边界
        u[-1, j + 1] = 0  # 右边界

    return u, x, t

if __name__ == "__main__":
    # 演示和测试
    params = {
        'a': 100,
        'L': 1,
        'd': 0.1,
        'C': 1,
        'sigma': 0.3,
        'dx': 0.01,
        'dt': 5e-5,
        'total_time': 0.1
    }
    u_sol, x_sol, t_sol = solve_wave_equation_ftcs(params)

    # 创建动画
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, xlim=(0, params['L']), ylim=(u_sol.min() * 1.1, u_sol.max() * 1.1))
    line, = ax.plot([], [], 'g-', lw=2)
    ax.set_title("1D Wave Equation (FTCS)")
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Displacement")

    def update(frame):
        line.set_data(x_sol, u_sol[:, frame])
        return line,

    # 调整动画速度
    interval = 1000 * params['dt'] * 100  # 乘以100加速显示
    ani = FuncAnimation(fig, update, frames=len(t_sol), interval=interval, blit=True)
    
    plt.tight_layout()
    plt.show()    
