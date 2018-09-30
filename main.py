import simple_dense

X = np.array([[0, 0, 0],
              [1, 0, 1],
              [0, 1, 1],
              [1, 1, 0],
              [0, 1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0],
              [1]])
print("Learning start")
net = simple_dense.network(X, y, 10, 10)
for j in range(100000):
    net.epoch()
    if j%1000 == 0:
        net.epoch(1)

net.epoch(1)
print("Learning end")

X = np.array([[0, 0, 0],
              [1, 0, 1],
              [0, 1, 1],
              [1, 1, 0]])
net.predict(X)
