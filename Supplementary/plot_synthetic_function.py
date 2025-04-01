# -------- --------------------------------------- -------- #
# -------- --------------------------------------- -------- #
# -------- This Script is NOT part of the pipeline -------- #
# -------- --------------------------------------- -------- #
# -------- --------------------------------------- -------- #

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(0) # random seed
x0 = np.linspace(0, 1, 100)
x1 = np.linspace(0, 10, 100)
X0, X1 = np.meshgrid(x0, x1)

epsilon = np.random.normal(0, 1, size=X0.shape)

# Compute target variable - y
Y = np.sin(2 * np.pi * X0) + np.log10(X1 + 1) + 0.5 * epsilon

# Plotting
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X0, X1, Y, cmap='viridis', edgecolor='none')

ax.set_title("3D Plot of y = sin(2πX₀) + log₁₀(X₁ + 1) + 0.5ε")
ax.set_xlabel("X₀")
ax.set_ylabel("X₁")
ax.set_zlabel("y")

fig.colorbar(surf, shrink=0.5, aspect=10)

plt.tight_layout()
plt.show()
