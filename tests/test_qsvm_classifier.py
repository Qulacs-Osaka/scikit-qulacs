import pandas as pd
from sklearn import datasets
from sklearn.model_selection import GridSearchCV, train_test_split

from skqulacs.circuit import create_ibm_embedding_circuit
from skqulacs.qsvm import QSVC

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
x = df.loc[:, ["petal length (cm)", "petal width (cm)"]]
x_train, x_test, y_train, y_test = train_test_split(
    x, iris.target, test_size=0.25, random_state=1
)

x_train = x_train.to_numpy()
x_test = x_test.to_numpy()
n_qubit = 4  # qubitの数 2だと少なすぎて複雑さがでない
circuit = create_ibm_embedding_circuit(n_qubit)

param_grid = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["linear", "poly", "rbf"],
    "gamma": ["scale", "auto"],
}

qsvm = QSVC(circuit)
grid_search = GridSearchCV(qsvm, param_grid, cv=5, n_jobs=-1)
grid_search.fit(x_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")

y_pred = grid_search.predict(x_test)
print(f"Test score: {grid_search.score(x_test, y_test)}")
