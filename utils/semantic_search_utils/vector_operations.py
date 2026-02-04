def add_vectors(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same length")
    
    return [a + b for a, b in zip(v1, v2)]

def subtract_vectors(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same length")
    
    return [a - b for a, b in zip(v1, v2)]


if __name__ == "__main__":
    v1 = [1, 2, 3]
    v2 = [4, 5, 6]
    print(add_vectors(v1, v2))
    print(subtract_vectors(v1, v2))