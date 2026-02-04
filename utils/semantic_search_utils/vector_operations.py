import numpy as np

def add_vectors(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same length")
    
    return np.add(v1, v2)

def subtract_vectors(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same length")
    
    return np.subtract(v1, v2)

def dot_product(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same length")
    
    return np.dot(v1, v2)

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


if __name__ == "__main__":
    v1 = [1, 2, 3]
    v2 = [4, 5, 6]
    print("Addition of v1 and v2:", add_vectors(v1, v2))
    print("Subtraction of v1 and v2:", subtract_vectors(v1, v2))
    print("Dot product of v1 and v2:", dot_product(v1, v2))
    print("Cosine similarity of v1 and v2:", cosine_similarity(v1, v2))