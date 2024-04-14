import pandas as pd
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
data = pd.read_csv("Modern_Periodic_Table.csv")
features = ['Atomic Mass', 'Density', 'Melting Point', 'Boiling Point', 'Heat of Fusion', 'Heat of Vaporization',
            'Specific Heat', 'Thermal Conductivity', 'Bulk Modulus', 'Young\'s Modulus']
data = data.dropna(subset=features)
component_data = data[features]
scaler = StandardScaler()
component_data_scaled = scaler.fit_transform(component_data)
iso = Isomap(n_components=2)
iso_data = iso.fit_transform(component_data_scaled)
plt.figure(figsize=(10, 8))
plt.scatter(iso_data[:, 0], iso_data[:, 1])
plt.title('Isomap Projection')
plt.xlabel('Component-1')
plt.ylabel('Component-2')
for i, txt in enumerate(data['Element Symbol']):
    plt.annotate(txt, (iso_data[i, 0], iso_data[i, 1]))
plt.show()
## ------------- LandMark ISOMAP (L-ISOMAP) ------------- ##
data2 = data.dropna(subset=features)
component_data2 = data[features]
scaler2 = StandardScaler()
component_data_scaled2 = scaler.fit_transform(component_data2)
num_landmarks = 10
landmark_indices = np.random.choice(len(component_data_scaled), num_landmarks, replace=False)
landmark_distances = np.zeros((num_landmarks, num_landmarks))
for i in range(num_landmarks):
    for j in range(i+1, num_landmarks):
        diff = component_data_scaled[landmark_indices[i]] - component_data_scaled[landmark_indices[j]]
        distance = np.sqrt(np.sum(diff ** 2))
        landmark_distances[i, j] = distance
        landmark_distances[j, i] = distance
iso_landmark2 = Isomap(n_neighbors=num_landmarks - 1, n_components=2, metric='precomputed')
iso_landmark.fit(landmark_distances)
data_landmark_distances = np.zeros((len(component_data_scaled), num_landmarks))
for i in range(len(component_data_scaled)):
    for j in range(num_landmarks):
        diff = component_data_scaled[i] - component_data_scaled[landmark_indices[j]]
        data_landmark_distances[i, j] = np.sqrt(np.sum(diff ** 2))
embedding_landmark = iso_landmark.transform(data_landmark_distances)
plt.figure(figsize=(10, 8))
plt.scatter(embedding_landmark[:, 0], embedding_landmark[:, 1])
plt.title('LandMark ISOMAP (L-ISOMAP)')
plt.xlabel('Component-1')
plt.ylabel('Component-2')
plt.show()
