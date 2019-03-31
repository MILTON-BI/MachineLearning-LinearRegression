import numpy as np
import tensorflow as tf

ls1 = [[1,2,3]]
vector1 = np.array(ls1)
print(vector1, vector1.shape)

ls2 = [[1,2,3],[4,5,6]]
matrix1 = np.array(ls2)
print(matrix1, matrix1.shape)

ls3 = [[1], [3], [5]]
vector2 = np.array(ls3)
print(vector2, vector2.shape)
#
res1 = vector1 * vector2
print(res1, res1.shape)
print(np.matmul(vector1,vector2))
#
# # ----------------------------------点乘和叉乘---------------------------------------------
# # 矩阵点乘(点积)：可以用* 或者用np.multiply()
# # 点积要求两个矩阵必须形状相同
# # 矩阵叉乘：np.matmul()
# # ----------------------------------矩阵（向量）转置---------------------------------------------
# # 方法1：矩阵(向量)名.T
# print(vector2.T)
# # 方法2：reshape()
# print(vector2.reshape(1,3))

# # ----------------------------------随机向量---------------------------------------------
print(tf.random_normal([12,1], stddev=0.01))