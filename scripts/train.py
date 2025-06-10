def train_pair(model, dataloader, criterion, optimizer, device, epochs=5):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for i, (text1, text2, label) in enumerate(dataloader):
            #text1, text2, label = text1.to(device), text2.to(device), label.to(device)
            label = label.to(device)

            # Forward pass
            z1, z2 = model(text1, text2)
            loss = criterion(z1, z2, label)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 100 == 0:
                print(f"Step {i} complete out of {len(dataloader)}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch Loss: {avg_loss:.4f}")
        return avg_loss

def train_triplet(model, dataloader, criterion, optimizer, device, epochs=5):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for i, (anchor_text, positive_text, negative_text) in enumerate(dataloader):
            # Forward pass
            z_anchor, z_positive, z_negative = model(anchor_text, positive_text, negative_text)

            loss = criterion(z_anchor, z_positive, z_negative)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 100 == 0:
                print(f"Step {i} complete out of {len(dataloader)}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch Loss: {avg_loss:.4f}")
        return avg_loss
    
################ Alternative Version
# def train_general(model, dataloader, criterion, optimizer, device, mode="pair", epochs=5):
#     """
#     General training function for both pair and triplet models.

#     Args:
#         model: the model to train
#         dataloader: a PyTorch DataLoader
#         criterion: a loss function
#         optimizer: optimizer instance
#         device: torch.device
#         mode: "pair" or "triplet"
#         epochs: number of training epochs
#     """
#     model.to(device)
#     model.train()

#     for epoch in range(epochs):
#         total_loss = 0.0
#         for i, batch in enumerate(dataloader):
#             optimizer.zero_grad()

#             if mode == "pair":
#                 text1, text2, label = batch
#                 label = label.to(device)
#                 z1, z2 = model(text1, text2)
#                 loss = criterion(z1, z2, label)

#             elif mode == "triplet":
#                 anchor_text, positive_text, negative_text = batch
#                 z_anchor, z_positive, z_negative = model(anchor_text, positive_text, negative_text)
#                 loss = criterion(z_anchor, z_positive, z_negative)

#             else:
#                 raise ValueError(f"Unsupported training mode: {mode}")

#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#             if i % 100 == 0:
#                 print(f"Step {i} complete out of {len(dataloader)}")

#         avg_loss = total_loss / len(dataloader)
#         print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

#     return avg_loss
