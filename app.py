import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load sample data
data = {
    "customers": [30, 45, 25, 60, 50, 55, 40, 70, 65, 80],
    "avg_spend": [15, 18, 14, 20, 19, 17, 16, 21, 22, 25],
    "rent_cost": [500, 600, 550, 700, 650, 620, 580, 750, 720, 800],
    "utilities": [100, 120, 110, 150, 130, 140, 125, 160, 155, 180],
    "profit": [400, 700, 300, 1000, 850, 900, 600, 1200, 1100, 1500],
}

df = pd.DataFrame(data)

# Prepare dataset
X = df.drop(columns=["profit"]).values
y = df["profit"].values.reshape(-1, 1)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Define the neural network
class ProfitPredictor(nn.Module):
    def __init__(self):
        super(ProfitPredictor, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Train model
model = ProfitPredictor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the model
torch.save(model.state_dict(), "models/profit_model.pth")

# Streamlit UI
st.title("Barbershop Profit Predictor")

customers = st.number_input("Number of Customers", min_value=0, value=50)
avg_spend = st.number_input("Average Spend per Customer", min_value=0, value=20)
rent_cost = st.number_input("Rent Cost", min_value=0, value=600)
utilities = st.number_input("Utilities Cost", min_value=0, value=150)

if st.button("Predict Profit"):
    input_data = np.array([[customers, avg_spend, rent_cost, utilities]])
    input_data = scaler_X.transform(input_data)
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    
    # Load the trained model and make predictions
    model.load_state_dict(torch.load("models/profit_model.pth"))
    model.eval()
    
    with torch.no_grad():
        predicted_profit = model(input_tensor)
    
    predicted_profit = scaler_y.inverse_transform(predicted_profit.numpy())
    st.write(f"Predicted Profit: ${predicted_profit[0][0]:.2f}")
