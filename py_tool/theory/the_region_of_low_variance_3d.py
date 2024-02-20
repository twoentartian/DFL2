import matplotlib.pyplot as plt
import numpy as np

# 3d
xs = np.linspace(-10,10,50)
ys = np.linspace(-10,10,50)
zs = np.linspace(-10,10,50)

x_plot = []
y_plot = []
z_plot = []
var_plot = []
for x in xs:
    for y in ys:
        for z in zs:
            x_plot.append(x)
            y_plot.append(y)
            z_plot.append(z)
            var_plot.append(np.var([x, y, z]))


# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
scat = ax.scatter(x_plot, y_plot, z_plot, c=var_plot, cmap='viridis', linewidth=0.5)

# Labels and title
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title('the variance')

# Color bar
fig.colorbar(scat, shrink=0.5, aspect=5)

plt.show()