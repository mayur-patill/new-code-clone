# new-code-clone
img is added 
1. Calculate the mean and standard deviation.

First.py

import numpy as np

data = [10, 20, 30, 40, 50]

mean_value = np.mean(data)

std_deviation = np.std(data)

print("Mean: ",mean_value)

print("Standard Deviation:", std_deviation)


3. Perform data filtering, and calculate aggregate statistics.

data.csv

Name Age Salary

Harshal 31 50000

Humanshu 26 40000

Tushar 29 60000

Shamkant 22 45000

Vikrant 20 70000

Third.py

import pandas as pd

# Load data

df = pd.read_csv('data.csv')

print("Original Data:")

print(df.head())

# Perform data filtering (Age > 25)

filtered_df = df[df['Age'] > 25]

# Calculate aggregate statistics

mean_age = filtered_df['Age'].mean()

median_salary = filtered_df['Salary'].median()

total_salary = filtered_df['Salary'].sum()

# Display results

print("\nFiltered Data (Age > 25):")

print(filtered_df)

print("\nAggregate Statistics:")

print("Mean Age:", mean_age)

print("Median Salary:", median_salary)

print("Total Salary Sum:", total_salary)




5. Implement the Clustering using K-means.

Fifth.py

import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

# Generate data

np.random.seed(0)

X = np.vstack([

 np.random.normal([0, 0], 0.5, (50, 2)),

 np.random.normal([5, 5], 1, (50, 2)),

 np.random.normal([10, 0], 1.5, (50, 2))

])

# K-means clustering and plot

kmeans = KMeans(n_clusters=3).fit(X)

plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')

plt.scatter(*kmeans.cluster_centers_.T, c='red', s=200)

plt.title('K-means Clustering')

plt.show()




6. Classification using Random Forest.

Sixth.py

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

# Load data and split

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, 

random_state=42)

# Train model and predict

clf = RandomForestClassifier(n_estimators=10).fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Print results

print("Accuracy: {:.2f}".format(accuracy_score(y_test, y_pred)))

print("Sample prediction:", iris.target_names[clf.predict([[5.1,3.5,1.4,0.2]])[0]])




7. Regression Analysis using Linear Regression.

data.csv

X Y

1 2

2 4

3 5

4 4

5 5

6 7

7 8

8 9

9 10

10 12

Seventh.py

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

# Load data from CSV file

df = pd.read_csv('data.csv')

# Splitting data into features and target variable

X = df[['X']]

y = df['Y']

# Splitting dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the model

model = LinearRegression()

model.fit(X_train, y_train)

# Making predictions

y_pred = model.predict(X_test)

# Evaluating the model

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

print("Intercept:", model.intercept_)

print("Coefficient:", model.coef_[0])

# Predicting for a new value

new_X = pd.DataFrame({'X': [11]}) # Ensure new input has feature names

predicted_Y = model.predict(new_X)

print("Predicted Y for X=11:", predicted_Y[0])

# Plotting Salary vs Experience

plt.scatter(X, y, color='blue', label='Actual Data')

plt.plot(X, model.predict(X), color='red', label='Regression Line')

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.title('Salary vs Experience')

plt.legend()

plt.show()



11. Use of degrees distribution of a network.

Eleventh.py

import networkx as nx

import matplotlib.pyplot as plt

# Create a random network

G = nx.erdos_renyi_graph(100, 0.05)

# Calculate degrees

degrees = dict(G.degree)

# Extract degree values

degree_values = list(degrees.values())

# Plot degree distribution

plt.hist(degree_values, bins=range(max(degree_values)+2), align='left', rwidth=0.8)

plt.xlabel('Degree')

plt.ylabel('Frequency')

plt.title('Degree Distribution')

plt.show()



12. Graph visualization of a network using maximum, minimum, median, first quartile and third

quartile.

Twelve.py

import networkx as nx

import matplotlib.pyplot as plt

import numpy as np

# Create and visualize the network

G = nx.Graph([(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)])

pos = nx.spring_layout(G)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)

nx.draw(G, pos, with_labels=True)

plt.title("Network Visualization")

# Generate data and plot box plot

data = np.random.randn(100)

plt.subplot(1, 2, 2)

plt.boxplot(data, vert=False)

plt.title("Box Plot")

plt.tight_layout()

plt.show()
