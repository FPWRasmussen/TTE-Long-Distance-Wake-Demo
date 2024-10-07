import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet5(nn.Module):
    def __init__(self, in_channels, out_channels, num_extra_features):
        super(UNet5, self).__init__()
        # Encoder
        self.encode1 = nn.Conv2d(in_channels, 256, 3, 2, 1)
        self.encode2 = nn.Conv2d(256, 256, 3, 2, 1)
        self.encode3 = nn.Conv2d(256, 256, 3, 2, 1)
        self.encode4 = nn.Conv2d(256, 256, 3, 2, 1)
        self.encode5 = nn.Conv2d(256, 256, 3, 2, 1)
        self.encode6 = nn.Conv2d(256, 256, 3, 2, 1)
        self.encode7 = nn.Conv2d(256, 256, 3, 2, 1)
        self.encode8 = nn.Conv2d(256, 256, 3, 2, 1)
        
        # Latent space
        self.fc_encode = nn.Linear(512 + num_extra_features, 512)
        
        # Decoder
        self.decode1 = nn.ConvTranspose2d(256, 256, 4, 2, 1)
        self.decode2 = nn.Conv2d(256+256, 256, 3, 1, 1)
        self.decode3 = nn.ConvTranspose2d(256, 256, 4, 2, 1)
        self.decode4 = nn.Conv2d(256+256, 256, 3, 1, 1)
        self.decode5 = nn.ConvTranspose2d(256, 256, 4, 2, 1)
        self.decode6 = nn.Conv2d(256+256, 256, 3, 1, 1)
        self.decode7 = nn.ConvTranspose2d(256, 256, 4, 2, 1)
        self.decode8 = nn.Conv2d(256+256, 256, 3, 1, 1)
        self.decode9 = nn.ConvTranspose2d(256, 256, 4, 2, 1)
        self.decode10 = nn.Conv2d(256+256, 256, 3, 1, 1)
        self.decode11 = nn.ConvTranspose2d(256, 256, 4, 2, 1)
        self.decode12 = nn.Conv2d(256+256, 256, 3, 1, 1)
        self.decode13 = nn.ConvTranspose2d(256, 256, 4, 2, 1)
        self.decode14 = nn.Conv2d(256+256, 256, 3, 1, 1)
        self.decode15 = nn.ConvTranspose2d(256, 256, 4, 2, 1)
        self.decode16 = nn.Conv2d(256, out_channels, 3, 1, 1)
        
        # Dropout
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, extra_vars):
        # Encoder
        enc1 = F.relu(self.encode1(x))
        enc2 = F.relu(self.encode2(enc1))
        enc3 = F.relu(self.encode3(enc2))
        enc4 = F.relu(self.encode4(enc3))
        enc5 = F.relu(self.encode5(enc4))
        enc6 = F.relu(self.encode6(enc5))
        enc7 = F.relu(self.encode7(enc6))
        x_mark = F.relu(self.encode8(enc7))

    
        x = x_mark.reshape(x_mark.size(0), -1)
        x = torch.cat((x, extra_vars), dim=1)
        x = self.dropout(F.relu(self.fc_encode(x)))
        x = x.reshape(x_mark.shape)
        
        # Decoder
        x = F.relu(self.decode1(x))
        x = torch.cat((x, enc7), dim=1)
        x = F.relu(self.decode2(x))
        x = F.relu(self.decode3(x))
        x = torch.cat((x, enc6), dim=1)
        x = F.relu(self.decode4(x))
        x = F.relu(self.decode5(x))
        x = torch.cat((x, enc5), dim=1)
        x = F.relu(self.decode6(x))
        x = F.relu(self.decode7(x))
        x = torch.cat((x, enc4), dim=1)
        x = F.relu(self.decode8(x))
        x = F.relu(self.decode9(x))
        x = torch.cat((x, enc3), dim=1)
        x = F.relu(self.decode10(x))
        x = F.relu(self.decode11(x))
        x = torch.cat((x, enc2), dim=1)
        x = F.relu(self.decode12(x))
        x = F.relu(self.decode13(x))
        x = torch.cat((x, enc1), dim=1)
        x = F.relu(self.decode14(x))
        x = F.relu(self.decode15(x))
        x = self.decode16(x)
        return x

    
if __name__ == "__main__":
    # Generate random data
    batch_size = 3
    num_channels = 3
    image_size = 256
    num_classes = 10
    num_extra_features = 4

    inputs = torch.randn(batch_size, num_channels, image_size*2, image_size)
    outputs = torch.randn(batch_size, num_classes, image_size*2, image_size)
    labels = torch.randn(batch_size,num_extra_features)

    # Instantiate the model
    model = UNet5(in_channels=num_channels, out_channels=num_classes, num_extra_features=num_extra_features)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


        # Count the total number of parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total number of parameters: {}".format(total_params))
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        model_outputs = model(inputs, labels)
        loss = criterion(model_outputs, outputs)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
