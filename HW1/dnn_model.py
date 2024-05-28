import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define the neural network model
class DNN_1Layer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DNN_1Layer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DNN_2Layer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DNN_2Layer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DNN_3Layer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(DNN_3Layer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.num_layers = num_layers

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.num_layers > 1:
            for i in range(1, self.num_layers):
                x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Function to initialize weights
def initialize_weights(m, init_type="xavier"):
    if isinstance(m, nn.Linear):
        if init_type == "xavier":
            nn.init.xavier_uniform_(m.weight)
        elif init_type == "he":
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        m.bias.data.fill_(0.01)


# Save the model
def save_model(model, path):
    torch.save(model.state_dict(), path)


# Load the model
def load_model(model, path):
    model.load_state_dict(torch.load(path))


def train(i, model, num_epochs, dataloader, size, learning_rate):
    print(size)

    # model.apply(
    #    lambda m: initialize_weights(m, init_type="xavier")
    # )  # Xavier initialization

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in dataloader:
            optimizer.zero_grad()
            images = images.reshape(-1, 28 * 28)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}"
        )

    # Save the model
    save_path = f"model_{i}_{size}.pth"
    save_model(model, save_path)


def test(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.reshape(-1, 28 * 28)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def trainP2(i, model, num_epochs, dataloader, size, learning_rate):
    print(size)

    # model.apply(
    #    lambda m: initialize_weights(m, init_type="xavier")
    # )  # Xavier initialization

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}"
        )

    # Save the model
    save_path = f"model_{i}_{size}.pth"
    save_model(model, save_path)


def testP2(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total
