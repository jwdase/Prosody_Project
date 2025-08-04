import torch
from torch import nn
from language_detection import config

def train_loop(model, train_loader, val_loader, base):
    """
    Sets up complete training loop
    """
    
    # ----------------------------- Defines All parameters

    model.to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), config.LR, weight_decay=config.DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode="min", 
        patience=config.PATIENCE, 
        factor=config.FACTOR, 
        threshold=config.THRESHOLD, 
    )

    # ----------------------------- Training Begins Below

    total_loss = []
    validation_loss = []
    validation_accuracy = []
    learning_rate = []
    best_acc = 0.0

    for i in range(config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        total_train = 0

        for inputs, lengths, labels in train_loader:

            inputs = inputs.to(config.DEVICE)
            lengths = lengths.to(config.DEVICE)
            labels = labels.to(config.DEVICE)


            # Shape [64, 1025, 172] --> [64, 1, 1025, 172]
            inputs = inputs.unsqueeze(1)

            # Forward Pass
            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)

            # Backprop + Optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Tracking loss
            running_loss += loss.item() * inputs.size(0)

            # Tracks total number of values in train
            total_train += labels.size(0)

        total_loss.append(running_loss / total_train)

        model.eval()
        running_val_loss = 0.0
        correct = 0
        total_val = 0

        with torch.no_grad():
            for inputs, lengths, labels in val_loader:
                inputs = inputs.to(config.DEVICE)
                lengths = lengths.to(config.DEVICE)
                labels = labels.to(config.DEVICE)

                # Shape [64, 1025, 172] --> [64, 1, 1025, 172]
                inputs = inputs.unsqueeze(1)

                # Calculates models predictions
                outputs = model(inputs, lengths)
                loss = criterion(outputs, labels)

                # Solves loss
                running_val_loss += loss.item() * inputs.size(0)

                # Checks correct entries
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

                # Tracks total number of values in validation
                total_val += labels.size(0)

        scheduler.step(running_val_loss)

        validation_loss.append(running_val_loss / total_val)
        validation_accuracy.append(correct / total_val)
        learning_rate.append(scheduler.get_last_lr())

        if validation_accuracy[-1] > best_acc + config.ERROR:
            best_acc = validation_accuracy[-1]
            torch.save(model.state_dict(), f"{base}/best_model.pth")


        if i % 2 == 0:
            print(f"Epoch [{i+1}/{config.NUM_EPOCHS}]")
            print(f"  Train loss:      {total_loss[-1]:.4f}")
            print(f"  Validation loss: {validation_loss[-1]:.4f}")
            print(f"  Validation acc:  {validation_accuracy[-1]:.4f}")

    # Deletes for cleanup
    del optimizer, scheduler

    return total_loss, validation_loss, learning_rate