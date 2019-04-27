from preprocess.processor import get_processed_data
from sklearn.decomposition import PCA
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt

(new_X, new_X_raw, new_y,
 max_t_s_num,
 num_student,
 test_new_X, test_new_y) = get_processed_data()

active_x = np.asarray(test_new_X)

result = PCA(n_components=3).fit_transform(active_x)

# df = DataFrame(random_rows(result, 100))
df = DataFrame(result)

graph = plt.figure().gca(projection='3d')
graph.scatter(df[0], df[1], df[2])
graph.set_xlabel('Feature 0')
graph.set_ylabel('Feature 1')
graph.set_zlabel('Feature 2')

plt.show()

result2d = PCA(n_components=2).fit_transform(active_x)
df2d = DataFrame(result2d)
graph2d = plt.scatter(df2d[0], df2d[1])
plt.show()
