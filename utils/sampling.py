# 重み付きサンプリングの実装
import random

grid = [10,10,10,10]
weight_sum = sum(grid)
print(weight_sum)

r = random.random() * weight_sum
print(r)

num = 0
for i, weight in enumerate(grid):
    num += weight
    if r <= num:
        print(i)
        break
