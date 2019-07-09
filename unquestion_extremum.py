'''
无条件情况：二元函数极值
知识点：
1、牛顿迭代法求f(x,y)驻点
2、数值微分递归求f(x,y)高阶偏导数
'''
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def fx(x):
    x1, x2 = x
    return (x1 + x2) / (x1**2 + x2**2 + 1)
'''数值微分：求fx在id方向上的deep阶偏导数'''
def dfx(x, id=0, h=0.001, deep=1):
    arr1, arr2 = x[:], x[:]
    arr1[id] = x[id] + h  # x在id方向上的变化量
    arr2[id] = x[id] - h
    if deep == 1:
        return (fx(x=arr1) - fx(x=arr2)) / (2 * h)
    else:
        return (dfx(x=arr1, id=id, h=h, deep=deep-1) - dfx(x=arr2, id=id, h=h, deep=deep-1)) / (2 * h)
'''牛顿迭代法：求fx在id方向上的驻点 -> dfx1=0，dfx2=0'''
def newtonIter(x0, id=0):
    while True:
        f1 = dfx(x=x0, id=id, deep=1)  # 一阶偏导数
        f22 = dfx(x=x0, id=id, deep=2)  # 二阶偏导数
        if f22 != 0:
            xk = x0[id] - f1 / f22  # 牛顿迭代，求一阶偏导数=0的驻点
            if np.abs(f1) < 1e-1:  # 一阶偏导数在误差范围内
                return xk
            x0[id] = xk  # 更新x0在id方向上的因变量
def solve(points):
    for x1, x2 in points:
        A = dfx(x=[x1, x2], id=0, deep=2)  # x1的二阶偏导数
        C = dfx(x=[x1, x2], id=1, deep=2)  # x2的二阶偏导数
        B = dfx(x=[dfx(x=[x1, x2], id=0, deep=1), x2], id=1, deep=1)  # x1和x2的一阶混合偏导数
        Hessian = np.mat([[A, B], [B, C]])  # 构造黑塞矩阵
        detH = np.linalg.det(Hessian)  # 计算黑塞矩阵的行列式=AC-B^2
        if detH > 0:
            print('Hessian为正定矩阵，f(x,y)在({},{})处取极{}值={}'.format(x1, x2, '大' if A < 0 else '小', fx(x=[x1, x2])))
        elif detH < 0:
            print('Hessian为不定矩阵，f(x,y)在({},{})处无极值'.format(x1, x2))
        else:
            print('Hessian为半定矩阵，f(x,y)在({},{})处无法判断极值'.format(x1, x2))
if __name__ == '__main__':
    # 1、求Fxy的驻点（牛顿迭代法求解F'xy=0）
    x1 = newtonIter(x0=[1, 1.2], id=0)
    x2 = newtonIter(x0=[-1, -1.2], id=0)
    y1 = newtonIter(x0=[1.2, 1], id=1)
    y2 = newtonIter(x0=[-1.2, -1], id=1)
    # 2、根据驻点计算黑塞矩阵，判断Fxy的极值情况
    solve([(x1, y1), (x1, y2), (x2, y1), (x2, y2)])
    # 函数曲面图像可视化
    fig = plt.figure()
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = (X + Y) / (X**2 + Y**2 + 1)
    # 函数图像
    # axes3d = Axes3D(fig)
    # axes3d.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    # 函数等值线
    # C = plt.contour(X, Y, Z, 10, colors='black')
    # plt.contourf(X, Y, Z, 10, cmap='rainbow')
    # plt.clabel(C, inline=True, fontsize=12)
    # axes3d.set_xlabel('X')
    # axes3d.set_ylabel('Y')
    # axes3d.set_zlabel('Z')
    # plt.show()
