
import matplotlib.pyplot as plt

cosine_deviations = [[2, 3, 1], [9, 8, 7], [5, 4, 6]]
x = []
y = []
for i in range(len(cosine_deviations)):
    for j in range(len(cosine_deviations[i])):
        x.append(i)
        y.append(cosine_deviations[i][j])
plt.scatter(x, y, label=f'1')


plt.legend()
plt.savefig("cosine_distances.png")
