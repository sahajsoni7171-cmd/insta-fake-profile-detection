import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from model import InstaFakeDetector

# Load dataset
df = pd.read_csv("train.csv")

# Features and target
X = df.drop(columns=['fake'])  # Input features
y = df['fake']                 # Labels (0 or 1)

# Split into training and test sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)

# Define model
input_dim = X.shape[1]
model = InstaFakeDetector(input_dim)

# Training config
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 50

# Training loop
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_val_tensor)
    predicted_labels = torch.argmax(predictions, dim=1)

print("\nClassification Report:")
print(classification_report(y_val_tensor, predicted_labels))

print("Confusion Matrix:")
print(confusion_matrix(y_val_tensor, predicted_labels))

# Save model
torch.save(model.state_dict(), "trained_model.pth")
print("\nModel saved as 'trained_model.pth'")
