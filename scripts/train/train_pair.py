def train_pair(model, dataloader, criterion, optimizer, device, epochs=5):
    model.to(device)
    model.train()

    best_epoch_loss = float('inf')
    for epoch in range(epochs):
        epoch_loss = 0.0
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

            epoch_loss += loss.item()

            if i % 100 == 0:
                print(f"Step {i} complete out of {len(dataloader)}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch Loss: {avg_loss:.4f}")

        if avg_loss < best_epoch_loss:
            best_epoch_loss = avg_loss

    return best_epoch_loss