# xuat thu vien
import numpy
from matplotlib import pyplot
from scipy.ndimage import gaussian_filter
import os
from numba import jit

# tham so matplotlib
pyplot.rcParams['font.family'] = 'serif'
pyplot.rcParams['font.size'] = 16

# image_out
output_dir = "C:/image_out1"
os.makedirs(output_dir, exist_ok=True)

# truong eta
@jit(nopython=True)
def update_eta_2D(eta, M, N, dx, dy, dt, nx, ny):
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            dMdx = (M[j, i + 1] - M[j, i - 1]) / (2. * dx)
            dNdy = (N[j + 1, i] - N[j - 1, i]) / (2. * dy)
            eta[j, i] = eta[j, i] - dt * (dMdx + dNdy)
    eta[0, :] = eta[1, :]
    eta[-1, :] = eta[-2, :]
    eta[:, 0] = eta[:, 1]
    eta[:, -1] = eta[:, -2]
    return eta

# truong m
@jit(nopython=True)
def update_M_2D(eta, M, N, D, g, h, alpha, dx, dy, dt, nx, ny):
    arg1 = M ** 2 / D
    arg2 = M * N / D
    fric = g * alpha ** 2 * M * numpy.sqrt(M ** 2 + N ** 2) / D ** (7. / 3.)
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            darg1dx = (arg1[j, i + 1] - arg1[j, i - 1]) / (2. * dx)
            darg2dy = (arg2[j + 1, i] - arg2[j - 1, i]) / (2. * dy)
            detadx = (eta[j, i + 1] - eta[j, i - 1]) / (2. * dx)
            M[j, i] = M[j, i] - dt * (darg1dx + darg2dy + g * D[j, i] * detadx + fric[j, i])
    return M

# truong n
@jit(nopython=True)
def update_N_2D(eta, M, N, D, g, h, alpha, dx, dy, dt, nx, ny):
    arg1 = M * N / D
    arg2 = N ** 2 / D
    fric = g * alpha ** 2 * N * numpy.sqrt(M ** 2 + N ** 2) / D ** (7. / 3.)
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            darg1dx = (arg1[j, i + 1] - arg1[j, i - 1]) / (2. * dx)
            darg2dy = (arg2[j + 1, i] - arg2[j - 1, i]) / (2. * dy)
            detady = (eta[j + 1, i] - eta[j - 1, i]) / (2. * dy)
            N[j, i] = N[j, i] - dt * (darg1dx + darg2dy + g * D[j, i] * detady + fric[j, i])
    return N

# 2d shallow water equation code 
def Shallow_water_2D(eta0, M0, N0, h, g, alpha, nt, dx, dy, dt, X, Y):
    eta = eta0.copy()
    M = M0.copy()
    N = N0.copy()
    D = eta + h
    ny, nx = eta.shape
    x = numpy.linspace(0, nx * dx, num=nx)
    y = numpy.linspace(0, ny * dy, num=ny)

    fig, ax = pyplot.subplots(figsize=(10, 6))
    cmap = 'seismic'
    extent = [numpy.min(x), numpy.max(x), numpy.min(y), numpy.max(y)]
    topo = ax.imshow(numpy.flipud(-h), cmap=pyplot.cm.gray, interpolation='nearest', extent=extent)
    im = ax.imshow(numpy.flipud(eta), extent=extent, interpolation='spline36', cmap=cmap, alpha=.75, vmin=-0.4, vmax=0.4)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    pyplot.colorbar(im, ax=ax, label=r'$\eta$ [m]')
    pyplot.colorbar(topo, ax=ax, label=r'$-h$ [m]')
    ax.invert_yaxis()

    nsnap = 50
    snap_count = 0

    for n in range(nt):
        eta = update_eta_2D(eta, M, N, dx, dy, dt, nx, ny)
        M = update_M_2D(eta, M, N, D, g, h, alpha, dx, dy, dt, nx, ny)
        N = update_N_2D(eta, M, N, D, g, h, alpha, dx, dy, dt, nx, ny)
        D = eta + h

        if (n % nsnap) == 0:
            im.set_data(numpy.flipud(eta))
            fig.canvas.draw()
            fig.canvas.flush_events()
            name_snap = os.path.join(output_dir, f"Shallow_water_2D_{snap_count + 1000}.tiff")
            pyplot.savefig(name_snap, format='tiff', bbox_inches='tight', dpi=125)
            snap_count += 1

    return eta, M, N
############################################################################################################################################
# setup cac gia tri dau vao
Lx = 100.0
Ly = 100.0
nx = 401
ny = 401
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
x = numpy.linspace(0.0, Lx, num=nx)
y = numpy.linspace(0.0, Ly, num=ny)
X, Y = numpy.meshgrid(x, y)
h = 50 * numpy.ones_like(X)
eta0 = 0.5 * numpy.exp(-((X - 50) ** 2 / 10) - ((Y - 50) ** 2 / 10))
M0 = 100. * eta0
N0 = 0. * M0
g = 9.81
alpha = 0.025
Tmax = 6.
dt = 1 / 4500.
nt = int(Tmax / dt)
############################################################################################################################################

# thuc hien tinh toan
eta, M, N = Shallow_water_2D(eta0, M0, N0, h, g, alpha, nt, dx, dy, dt, X, Y)