import numpy as np
# fit crash probability to color scale
# color 0 is red, 130 is green, middle is yellow
x = np.array([0.0, 0.05, 0.2, 0.5, 1])  # crash prob
y = np.array([130, 100, 60, 20, 1]) # color 
z = np.polyfit(x, y, 2) #coefficients, highest power first
p = np.poly1d(z)
print(p(0))
print(p(0.05))
print(p(0.2))
print(p(0.5))
print(p(1))
print(z)