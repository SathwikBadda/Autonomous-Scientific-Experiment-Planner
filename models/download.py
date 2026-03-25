from sentence_transformers import SentenceTransformer

# Model name
model_name = "all-MiniLM-L6-v2"

# Load model (downloads automatically first time)
model = SentenceTransformer(model_name)

# Save locally
save_path = "./models/all-MiniLM-L6-v2"
model.save(save_path)

print(f"Model saved at: {save_path}")