import numpy as np
import matplotlib.pyplot as plt

# Parameters
Lx = 1.0     # Length in x-direction
Ly = 1.0     # Length in y-direction
nx = 20      # Number of grid points in x-direction
ny = 20      # Number of grid points in y-direction
alpha = 0.01 # Thermal diffusivity
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
dt = 0.0005  # Time step size
nt = 500     # Number of time steps

# Initialize the temperature grid
u = np.zeros((nx, ny))
u_new = np.zeros((nx, ny))

# Boundary conditions
u[:, 0] = 100.0  # Left boundary
u[:, -1] = 100.0 # Right boundary
u[0, :] = 0.0    # Top boundary
u[-1, :] = 0.0   # Bottom boundary

# Finite difference method
for n in range(nt):
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            u_new[i, j] = u[i, j] + alpha * dt * (
                (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / dx**2 +
                (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]) / dy**2
            )

    # Update the temperature grid
    u = u_new.copy()

# Plot the final temperature distribution
plt.imshow(u, cmap='hot', origin='lower', extent=[0, Lx, 0, Ly])
plt.colorbar(label='Temperature')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Heat Conduction using Finite Difference Method')
plt.show()
