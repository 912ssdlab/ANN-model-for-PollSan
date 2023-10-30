import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import  DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor


from sklearn.metrics import accuracy_score
import time

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
def genertate(lenth=10):
    gc_list = [0, 1]
    wl_list = [0, 1]
    write_list = list(np.linspace(0, 1, lenth, dtype=float))
    invalid_list = list(np.linspace(0, 1, lenth, dtype=float))
    invalid_list =  np.around(invalid_list, 1)
    write_list = np.around(write_list,1)
    gc_df = pd.DataFrame({"X_GC": gc_list})
    wl_df = pd.DataFrame({"X_Wear": wl_list})
    write_df = pd.DataFrame({"X_Wr": write_list})
    invalid_df = pd.DataFrame({"X_Inv": invalid_list})
    gc_df['value'] = 1
    wl_df['value'] = 1
    write_df['value'] = 1
    invalid_df['value'] = 1
    df3 = gc_df.merge(wl_df, how='left', on='value')
    df4 = df3.merge(write_df, how='left', on='value')
    df5 = df4.merge(invalid_df, how='left', on='value')
    return df5.drop('value',axis=1)
data = pd.read_excel("data_set.xlsx")
array = data.values
scalar = MinMaxScaler()
array = scalar.fit_transform(array)
X = array[:, :4]
Y = array[:, 4:5]
validation_size = 0.2
seed = 10
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,test_size=validation_size, random_state=seed)

genertate_data = genertate(lenth=10)
max_socres = 0
bestY = []
for i in range(1):
    mlp = MLPRegressor(max_iter=200,hidden_layer_sizes=(128,128,128),batch_size='auto',early_stopping=False,n_iter_no_change=200)
    start = time.time()
    mlp.fit(X_train, Y_train)
    time_used = time.time() - start
    print(time_used)
    K_pred = mlp.predict(X_validation)
    score =r2_score(Y_validation, K_pred)
    if (score > max_socres):
        max_socres = score
        genertated_pred = mlp.predict(np.array(genertate_data))
        loss_history = mlp.loss_curve_
        bestY = genertated_pred

# loss function
dic = {"loss": loss_history}
df = pd.DataFrame(dic)
df.to_excel("loss.xlsx")
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(loss_history)+1), loss_history)
plt.title('MLP Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
print("best_accuracy: ")
print(max_socres)
print(len(bestY))
bestY = normalization(bestY)
bestY = np.around(bestY,3)
genertate_data["Y"] = bestY
genertate_data.to_excel("index_table.xlsx")

