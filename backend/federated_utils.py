    # ==========================================
# federated_utils.py
# Federated Learning Utilities
# ==========================================

import copy
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import config
from secure_aggregation import secure_aggregate


# ==========================================
# 1️⃣ Local Training (Client Side)
# ==========================================

def local_train(model, dataloader, device=config.DEVICE):

    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE
    )

    total_loss = 0

    for epoch in range(config.LOCAL_EPOCHS):

        for images, metadata, labels in dataloader:

            images = images.to(device)
            metadata = metadata.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images, metadata)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    return model.state_dict(), total_loss / len(dataloader)


# ==========================================
# 2️⃣ Federated Averaging (Server Side)
# ==========================================

def federated_average(client_weights):

    global_weights = copy.deepcopy(client_weights[0])

    for key in global_weights.keys():
        for i in range(1, len(client_weights)):
            global_weights[key] += client_weights[i][key]

        global_weights[key] = torch.div(
            global_weights[key],
            len(client_weights)
        )

    return global_weights


# ==========================================
# 3️⃣ Evaluation Function
# ==========================================

def evaluate_model(model, dataloader, device=config.DEVICE):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, metadata, labels in dataloader:

            images = images.to(device)
            metadata = metadata.to(device)
            labels = labels.to(device)

            outputs = model(images, metadata)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm
    }


# ==========================================
# 4️⃣ Federated Training Loop (No DP)
# ==========================================

def federated_training(global_model,
                       client_loaders,
                       test_loader,
                       num_rounds=config.NUM_ROUNDS,
                       device=config.DEVICE):

    history = []

    for round_num in range(num_rounds):

        print(f"\n🔵 Federated Round {round_num+1}/{num_rounds}")

        client_weights = []
        round_losses = []

        # -------------------------
        # Each Client Trains Locally
        # -------------------------

        for i, loader in enumerate(client_loaders):

            print(f"Client {i+1} training...")

            client_model = copy.deepcopy(global_model)
            client_model.to(device)

            weights, loss = local_train(client_model, loader, device)

            client_weights.append(weights)
            round_losses.append(loss)

            print(f"Client {i+1} Loss: {loss:.4f}")

        # -------------------------
        # Server Aggregation
        # -------------------------

        if config.USE_SECURE_AGGREGATION:
            print("  🔐 Using secure aggregation ...")
            new_global_weights = secure_aggregate(
                client_weights, round_num=round_num,
                device=str(device)
            )
        else:
            new_global_weights = federated_average(client_weights)

        global_model.load_state_dict(new_global_weights)

        # -------------------------
        # Evaluate on Global Test
        # -------------------------

        metrics = evaluate_model(global_model, test_loader, device)

        print(f"Round {round_num+1} Accuracy: {metrics['accuracy']:.4f}")
        print(f"Round {round_num+1} F1-score: {metrics['f1']:.4f}")

        history.append({
            "round": round_num+1,
            "loss": sum(round_losses) / len(round_losses),
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"]
        })

    return global_model, history