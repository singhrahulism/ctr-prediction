import pickle

def save_artifacts(model, metadata, path):
    with open(path, "wb") as f:
        pickle.dump({
            "model": model,
            "metadata": metadata
        }, f)
    
def load_artifacts(path):
    with open(path, "rb") as f:
        return pickle.load(f)
