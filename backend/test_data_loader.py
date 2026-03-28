from data_loader import create_dataloaders

client1_loader, client2_loader, global_test_loader = create_dataloaders()

print("Client 1 batches:", len(client1_loader))
print("Client 2 batches:", len(client2_loader))
print("Global Test batches:", len(global_test_loader))

# Check one batch
images, metadata, labels = next(iter(client1_loader))

print("Image shape:", images.shape)
print("Metadata shape:", metadata.shape)
print("Labels shape:", labels.shape)