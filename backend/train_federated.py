# ==========================================
# train_federated.py
# Main Federated Training Script
# with Secure Aggregation + DP + Visualization
# ==========================================

import copy
import torch
import torch.optim as optim

import config
from data_loader import create_dataloaders
from backend.model import EfficientNet_ViT_Metadata, freeze_backbone
from federated_utils import federated_training, evaluate_model
from dp_utils import make_model_private, get_epsilon
from secure_aggregation import secure_aggregate
from visualize import (
    plot_training_history,
    plot_confusion_matrix,
    plot_per_class_accuracy,
    plot_comparison,
)


# ==========================================
# MAIN
# ==========================================

def main():

    print("🚀 Starting Federated Multimodal Training")
    print(f"   Secure Aggregation : {'ON' if config.USE_SECURE_AGGREGATION else 'OFF'}")
    print(f"   Differential Privacy: {'ON' if config.USE_DP else 'OFF'}")
    print(f"   Clients            : {config.NUM_CLIENTS}")
    print(f"   Rounds             : {config.NUM_ROUNDS}")
    print(f"   Device             : {config.DEVICE}")

    device = config.DEVICE

    # --------------------------------------
    # 1️⃣ Load Federated Dataset
    # --------------------------------------

    client1_loader, client2_loader, global_test_loader = create_dataloaders()
    client_loaders = [client1_loader, client2_loader]

    # --------------------------------------
    # 2️⃣ Load Centralized Pretrained Model
    # --------------------------------------

    global_model = EfficientNet_ViT_Metadata()
    global_model.load_state_dict(
        torch.load(config.CENTRALIZED_MODEL_PATH, map_location=device)
    )

    global_model.to(device)
    print("✅ Loaded centralized model")

    # --------------------------------------
    # 3️⃣ Freeze Backbone
    # --------------------------------------

    global_model = freeze_backbone(global_model)
    print("✅ Backbone frozen (only gate + classifier train)")

    # ======================================
    # 🔵 Phase 1: Federated Learning (No DP)
    #    Secure aggregation handled inside
    #    federated_training() via config flag
    # ======================================

    print("\n==============================")
    print("🔵 Running FL (No DP)")
    if config.USE_SECURE_AGGREGATION:
        print("   🔐 Secure Aggregation ENABLED")
    print("==============================")

    fl_model = copy.deepcopy(global_model)

    fl_model, fl_history = federated_training(
        fl_model,
        client_loaders,
        global_test_loader,
        num_rounds=config.NUM_ROUNDS,
        device=device
    )

    final_fl_metrics = evaluate_model(fl_model, global_test_loader, device)

    # Plot FL history + confusion matrix
    plot_training_history(fl_history, tag="FL")
    plot_confusion_matrix(final_fl_metrics["confusion_matrix"], tag="FL")
    plot_per_class_accuracy(final_fl_metrics["confusion_matrix"], tag="FL")

    # ======================================
    # 🔐 Phase 2: Federated Learning + DP
    # ======================================

    print("\n==============================")
    print("🔐 Running FL + Differential Privacy")
    if config.USE_SECURE_AGGREGATION:
        print("   🔐 Secure Aggregation ENABLED")
    print("==============================")

    dp_model = copy.deepcopy(global_model)

    epsilon_values = []
    dp_history = []

    for round_num in range(config.NUM_ROUNDS):

        print(f"\n🔵 DP Federated Round {round_num+1}/{config.NUM_ROUNDS}")

        client_weights = []

        for i, loader in enumerate(client_loaders):

            print(f"Client {i+1} DP training...")

            # ----------------------------------
            # 1️⃣ Copy global model
            # ----------------------------------
            client_model = copy.deepcopy(dp_model)
            client_model.to(device)
            client_model.train()

            # ----------------------------------
            # 2️⃣ Fresh optimizer
            # ----------------------------------
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, client_model.parameters()),
                lr=config.LEARNING_RATE
            )

            # ----------------------------------
            # 3️⃣ Attach Privacy Engine
            # ----------------------------------
            client_model, optimizer, private_loader, privacy_engine = make_model_private(
                client_model,
                optimizer,
                loader
            )

            # ----------------------------------
            # 4️⃣ Train locally with DP
            # ----------------------------------
            total_loss = 0.0
            num_batches = 0
            for images, metadata, labels in private_loader:

                images = images.to(device)
                metadata = metadata.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = client_model(images, metadata)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            # ----------------------------------
            # 5️⃣ Get epsilon
            # ----------------------------------
            epsilon = get_epsilon(privacy_engine)
            epsilon_values.append(epsilon)

            print(f"Client {i+1} ε = {epsilon:.4f}")

            # Unwrap DP model before aggregation
            client_weights.append(client_model._module.state_dict())

        # ----------------------------------
        # 6️⃣ Aggregation (Secure or Plain)
        # ----------------------------------

        if config.USE_SECURE_AGGREGATION:
            print("  🔐 Using secure aggregation ...")
            new_global_weights = secure_aggregate(
                client_weights, round_num=round_num,
                device=str(device)
            )
        else:
            new_global_weights = {}
            for key in client_weights[0].keys():
                new_global_weights[key] = sum(
                    client_weights[j][key] for j in range(len(client_weights))
                ) / len(client_weights)

        dp_model.load_state_dict(new_global_weights)

        # ----------------------------------
        # 7️⃣ Evaluate Global Model
        # ----------------------------------

        metrics = evaluate_model(dp_model, global_test_loader, device)

        print(f"Round {round_num+1} Accuracy: {metrics['accuracy']:.4f}")
        print(f"Round {round_num+1} F1-score: {metrics['f1']:.4f}")

        dp_history.append({
            "round": round_num + 1,
            "loss": total_loss / max(num_batches, 1),
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        })

    final_dp_metrics = evaluate_model(dp_model, global_test_loader, device)

    # Plot DP history + confusion matrix
    plot_training_history(dp_history, tag="FL_DP")
    plot_confusion_matrix(final_dp_metrics["confusion_matrix"], tag="FL_DP")
    plot_per_class_accuracy(final_dp_metrics["confusion_matrix"], tag="FL_DP")

    # Plot comparison
    plot_comparison(final_fl_metrics, final_dp_metrics)

    # ======================================
    # 📊 Final Comparison
    # ======================================

    print("\n==============================")
    print("📊 FINAL COMPARISON")
    print("==============================")

    print("\nCentralized Model Accuracy: 0.8600")

    print("\nFL (No DP)")
    print(f"Accuracy : {final_fl_metrics['accuracy']:.4f}")
    print(f"F1 Score : {final_fl_metrics['f1']:.4f}")

    print("\nFL + DP")
    print(f"Accuracy : {final_dp_metrics['accuracy']:.4f}")
    print(f"F1 Score : {final_dp_metrics['f1']:.4f}")
    print(f"Final ε (worst case): {max(epsilon_values):.4f}")

    # ===============================
    # SAVE FINAL MODELS
    # ===============================

    torch.save(fl_model.state_dict(), "final_FL_model.pth")
    print("✅ Final FL model saved.")

    torch.save(dp_model.state_dict(), "final_DP_model.pth")
    print("✅ Final DP model saved.")

    print("\n✅ All plots saved in results/ directory.")


if __name__ == "__main__":
    main()