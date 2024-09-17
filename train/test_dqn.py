import matplotlib
import matplotlib.pyplot as plt

# Check which backend matplotlib is using
print("Current backend:", matplotlib.get_backend())

# Generate a simple plot
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Test Plot')

# Show the plot
plt.show()






#THIS FILE WAS USED TO TEST THE PLOT NOT NECCESSARY