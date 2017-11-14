import numpy as np


def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))


class network:
    def __init__(self, X, y, length, height):
        self.X = X
        self.y = y
        self.syn = []
        self.result = 0
        self.syn.append(2 * np.random.random((len(X[0]), height)) - 1)
        for i in range(length):
            self.syn.append(2 * np.random.random((height, height)) - 1)
        self.syn.append(2 * np.random.random((height, len(y[0]))) - 1)

    def test(self, X):
        a = []
        a.append(X)
        for i in range(1, len(self.syn) + 1, 1):
            a.append(nonlin(np.dot(a[i - 1], self.syn[i - 1])))

        print("Inputs: ")
        print(X)
        print("Answers: ")
        print(a[len(a) - 1])
        print("\n")

    def epoch(self, show=0):
        a = []
        a.append(self.X)
        for i in range(1, len(self.syn) + 1, 1):
            a.append(nonlin(np.dot(a[i - 1], self.syn[i - 1])))
        error = [np.array([0])] * len(a)
        delta = [np.array([0])] * len(a)
        # Вычисляем последнию ошибку
        error[len(a) - 1] = self.y - a[len(a) - 1]
        delta[len(a) - 1] = error[len(a) - 1] * nonlin(a[len(a) - 1], deriv=True)

        if show == 3:
            print("Error:" + str(np.mean(np.abs(error[len(a) - 1]))))

        for i in range(len(a) - 2, 0, -1):
            error[i] = delta[i + 1].dot(self.syn[i].T)
            delta[i] = error[i] * nonlin(a[i], deriv=True)

        for i in range(len(self.syn)):
            self.syn[i] += a[i].T.dot(delta[i + 1])
        if show == 1:
            print("Inputs: ")
            print(self.X)
            print("Answers: ")
            print(a[len(a) - 1])
            print("\n")


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
print("Teaching start")
net = network(X, y, 10, 10)
for j in range(100000):
    net.epoch()
    if j%1000 == 0:
        net.epoch(3)

net.epoch(1)

print("Teaching end")

X = np.array([[0, 0, 0],
              [1, 0, 1],
              [0, 1, 1],
              [1, 1, 0]])
net.test(X)
