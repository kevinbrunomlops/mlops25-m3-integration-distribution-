import torch
import torch.nn as nn

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0
    correct = 0
    loss_sum = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    
    return {"loss": loss_sum / total, "acc": correct / total}

@torch.no_grad()
def eval_loop(model, loader, criterion, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        loss_sum += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    
    return {"loss": loss_sum / total, "acc": correct / total}

def fit(model, train_loader, test_loader, epochs, lr, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_metrics = eval_loop(model, test_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["acc"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {row['train_loss']:.4f} acc {row['train_acc']:.3f} | "
            f"test loss {row['test_loss']:.4f} acc {row['test_acc']:.3f}"
        )
    return history