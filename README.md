## Polling Sanitization to Balance I/O Latency and Data Security of High-density SSDs

### 1. Requirements

 * python 3.8
 * scikit-learn 1.1.1
 * pandas 1.4.2

### 2. Model
```MLPRegression.py```: The relevant code and parameters for the ANN model.

### 3. Data
```data_set.xlsx```: The dataset for model training. X_GC indicates whether there are pages with garbage collection. X_Wear indicates whether there are pages with wear leveling.
X_Wr indicates the normalized number of occurred write requests. X_Inv indicates the normalized number of occurred invalid pages.

### 4. Output
```index_table.xlsx```: The index table is generated by the pre-trained ANN model. It is used for invalid page prediction.
