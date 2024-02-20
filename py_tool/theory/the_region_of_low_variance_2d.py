import matplotlib.pyplot as plt
import numpy as np

# 2d
xs = np.linspace(-10,10,50)
ys = np.linspace(-10,10,50)

x_plot = []
y_plot = []
var_plot = []
for x in xs:
    for y in ys:
        x_plot.append(x)
        y_plot.append(y)
        var_plot.append(np.var([x, y]))


# Plot
fig = plt.figure()
ax = fig.add_subplot(111)

# Scatter plot
scat = ax.scatter(x_plot, y_plot, c=var_plot, cmap='viridis', linewidth=0.5)

# Labels and title
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.title('the variance')

# Color bar
fig.colorbar(scat, shrink=0.5, aspect=5)

plt.show()