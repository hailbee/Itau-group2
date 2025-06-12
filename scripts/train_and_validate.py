from test import test_model
from sklearn.metrics import precision_score, recall_score

### High number of epochs: 30 for now
def train_and_val_pair(model, dataloader, criterion, optimizer, device, test_reference_filepath, test_filepath, epochs=30):
    best_metrics = {
    'accuracy': 0.0,
    'precision': 0.0,
    'recall': 0.0
    }
    best_epochs = {
        'accuracy': -1,
        'precision': -1,
        'recall': -1
    }

    
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

        ### Evaluate on test set
        model.eval()
        results_df = test_model(model, test_reference_filepath, test_filepath)
       
        ### change this in line with ROC_AUC
        ### find the youden threshold first (use function from eval)
        ### make preds based on that
        preds = [1 if r > 0.8 else 0 for r in results_df['max_similarity']]  # Adjust threshold as needed
        labels = results_df['label'].tolist()

        acc = sum(p == l for p, l in zip(preds, labels)) / len(preds)
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)

        print(f"Epoch {epoch+1} - Test Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

        ### Track Bests
        if acc > best_metrics['accuracy']:
            best_metrics['accuracy'] = acc
            best_epochs['accuracy'] = epoch + 1
        if precision > best_metrics['precision']:
            best_metrics['precision'] = precision
            best_epochs['precision'] = epoch + 1
        if recall > best_metrics['recall']:
            best_metrics['recall'] = recall
            best_epochs['recall'] = epoch + 1

        model.train()

    ### Print Summary
    print("\n=== Best Epochs Summary ===")
    print(f"Best Accuracy: {best_metrics['accuracy']:.4f} at epoch {best_epochs['accuracy']}")
    print(f"Best Precision: {best_metrics['precision']:.4f} at epoch {best_epochs['precision']}")
    print(f"Best Recall: {best_metrics['recall']:.4f} at epoch {best_epochs['recall']}")

    return best_epoch_loss

def train_and_val_triplet(model, dataloader, criterion, optimizer, device, test_reference_filepath, test_filepath, epochs=30):
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
        
        ### Evaluate accuracy on test set
        model.eval()
        results_df = test_model(model, test_reference_filepath, test_filepath)
        preds = [1 if r > 0.8 else 0 for r in results_df['max_similarity']]  # Adjust threshold as needed
        labels = results_df['label'].tolist()

        acc = sum(p == l for p, l in zip(preds, labels)) / len(preds)
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)

        print(f"Test Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    
        model.train()

    return best_epoch_loss