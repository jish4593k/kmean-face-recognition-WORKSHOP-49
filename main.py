import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import mysql.connector
from sklearn.cluster import KMeans

# Fetch data from MySQL
mydb = mysql.connector.connect(
    host="localhost",
    user="login",
    passwd="pass",
    database="Db_Name"
)
mycursor = mydb.cursor()

mycursor.execute("SELECT face_data FROM facedata")
face_data = mycursor.fetchall()

mycursor.execute("SELECT name FROM facedata")
names = mycursor.fetchall()

# Convert face data to feature vectors
def getVectors(list_data):
    feature_vectors = []
    for data in list_data:
        data = data[0].split(";")
        feature_vector = [float(val) for val in data if val]
        feature_vectors.append(feature_vector)
    return feature_vectors

data = getVectors(face_data)
X = np.array(data)

# Apply StandardScaler for data scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform K-means clustering for comparison
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(X_scaled)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Use PCA for dimensionality reduction and visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
centers_pca = pca.transform(centers)

# Build a neural network for clustering
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(5, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Convert labels to one-hot encoding
labels_onehot = to_categorical(labels)

# Train the model
model.fit(X_scaled, labels_onehot, epochs=50, verbose=0)

# Predict the cluster label for new face data
new_data = np.array(getVectors(["-0.188544;0.095237;0.074243;-0.025494;...;0.109958;-0.023378;0.185327;0.053302;"]))
new_data_scaled = scaler.transform(new_data)
predicted_label = np.argmax(model.predict(new_data_scaled), axis=-1)

# Display the predicted cluster label
print("Predicted Label:", predicted_label[0])

# Plot the clustered data
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=20)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=100)
plt.title("Face Data Clustering (K-means)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

# Plot the original data and their labels
plt.subplot(2, 2, 2)
plt.xlim(-0.4, 0.4)
plt.ylim(-0.4, 0.4)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=7)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=100)
plt.title("Face Data Clustering (Original)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Visualize the training loss
history = model.history.history
loss = history['loss']
epochs = range(1, len(loss) + 1)
plt.subplot(2, 2, 3)
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")

# Add more functionality, such as saving the model and results
model.save("face_clustering_model.h5")
results = pd.DataFrame({'Names': [name[0] for name in names], 'Cluster Labels': labels})
results.to_csv("face_clustering_results.csv")

# Show the plots
plt.tight_layout()
plt.show()
