import pickle  # Importing the pickle library for data serialization
from sklearn.ensemble import RandomForestClassifier  # Importing RandomForestClassifier from scikit-learn
from sklearn.model_selection import train_test_split  # Importing train_test_split for splitting the dataset
from sklearn.metrics import accuracy_score  # Importing accuracy_score for evaluating the model
import numpy as np  # Importing NumPy for numerical operations

# Load the data from the pickle file
data_dict = pickle.load(open('data_BSL.pickle', 'rb'))

# Uncomment the following lines to inspect the data
# print(f"Keys: {data_dict.keys()}")
# print(f"First element: {data_dict['data'][0]}")
# Calculate the maximum sequence length in the data
max_length = max(len(seq) for seq in data_dict['data'])
# print(f"Maximum sequence length: {max_length}")

# Function to pad sequences to the same length
def pad_sequences(sequences, maxlen):
    padded_sequences = np.zeros((len(sequences), maxlen))  # Initialize an array of zeros
    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq)] = seq  # Copy the sequence to the corresponding row
    return padded_sequences

# Pad the data to make all sequences of the same length
data = pad_sequences(data_dict['data'], max_length)
labels = np.asarray(data_dict['labels'])  # Convert labels to a NumPy array

# Split the dataset into training and test sets
# Stratify ensures the same proportion of labels in train and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize the RandomForestClassifier model
model = RandomForestClassifier()
# Fit the model on the training data
model.fit(X_train, y_train)
# Predict the labels on the test data
y_predict = model.predict(X_test)
# Calculate the accuracy of the model
score = accuracy_score(y_predict, y_test)
print(score * 100)  # Print the accuracy score as a percentage

# Save the trained model to a pickle file
f = open('model_BSL.pkl', 'wb')
pickle.dump({'model': model}, f)
f.close()
