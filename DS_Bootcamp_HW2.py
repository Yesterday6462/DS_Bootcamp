### Part 1 ###
import numpy as np

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
print(iris_2d)

# 1. Define two custom numpy arrays, say A and B. Generate two new numpy arrays by stacking A and B vertically and horizontally.
A = iris_2d[:,0]
B = iris_2d[:,1]
print(np.vstack((A,B)))
print(np.hstack((A,B)))

# 2. Find common elements between A and B. [Hint : Intersection of two sets]
common_elements = np.intersect1d(A, B)
print(common_elements)

# 3. Extract all numbers from A which are within a specific range. eg between 5 and 10. [Hint: np.where() might be useful or boolean masks]
A_specific = A[np.where((A >= 5) & (A <= 10))]
print(A_specific)

# 4. Filter the rows of iris_2d that has petallength (3rd column) > 1.5 and sepallength (1st column) < 5.0
iris_2d_filtered = iris_2d[np.where((iris_2d[:,2] > 1.5) & (iris_2d[:,0] < 5))]
print(iris_2d_filtered)


### Part 2 ###
import pandas as pd

# 1. From df filter the 'Manufacturer', 'Model' and 'Type' for every 20th row starting from 1st (row 0).
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
print(df.head(10))
df_selected = df[::20]
df_filtered = df_selected[['Manufacturer', 'Model', 'Type']]
print(df_filtered)

# 2. Replace missing values in Min.Price and Max.Price columns with their respective mean.
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
print(df[['Min.Price', 'Max.Price']].head(10))
df['Min.Price'] = df['Min.Price'].fillna(df['Min.Price'].mean())
df['Max.Price'] = df['Max.Price'].fillna(df['Max.Price'].mean())
print(df[['Min.Price', 'Max.Price']].head(10))

# 3. How to get the rows of a dataframe with row sum > 100?
df = pd.DataFrame(np.random.randint(10, 40, 60).reshape(-1, 4))
print(df.head(10))
df_filtered = df[df.sum(axis=1) > 100]
print(df_filtered.head(10))