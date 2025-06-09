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

def train_triplet_e(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()

        anchor_input = {k: v.to(device) for k, v in batch['anchor'].items()}
        positive_input = {k: v.to(device) for k, v in batch['positive'].items()}
        negative_input = {k: v.to(device) for k, v in batch['negative'].items()}

        anchor_emb = model(**anchor_input)
        positive_emb = model(**positive_input)
        negative_emb = model(**negative_input)

        loss = criterion(anchor_emb, positive_emb, negative_emb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
