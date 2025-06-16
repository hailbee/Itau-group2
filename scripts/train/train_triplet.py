def train_triplet(model, dataloader, criterion, optimizer, device, epochs=5):
    model.to(device)
    model.train()

    best_epoch_loss = float('inf')
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, (anchor_text, positive_text, negative_text) in enumerate(dataloader):
            # Forward pass
            z_anchor, z_positive, z_negative = model(anchor_text, positive_text, negative_text)

            loss = criterion(z_anchor, z_positive, z_negative)

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

 
def train_triplet_warmup(model, warmup_loader, hard_loader, criterion, optimizer, device, warmup_epochs=5, epochs=10):
    model.to(device)
    model.train()

    best_epoch_loss = float('inf')
    for epoch in range(epochs):

        dataloader = warmup_loader if epoch < warmup_epochs else hard_loader
        epoch_loss = 0.0
        for i, (anchor_text, positive_text, negative_text) in enumerate(dataloader):
            # Forward pass
            z_anchor, z_positive, z_negative = model(anchor_text, positive_text, negative_text)

            loss = criterion(z_anchor, z_positive, z_negative)

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