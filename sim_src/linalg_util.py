import numpy as np


def generate_random_rotation_matrix(k):
    # Create a random matrix
    H = np.random.randn(k, k)

    # Perform QR decomposition
    Q, R = np.linalg.qr(H)

    # Ensure a proper rotation matrix (det(Q) should be 1)
    Q *= np.sign(np.linalg.det(Q))

    return Q
def generate_rand_regular_simplex_with_Z_vertices(Z,D=None):
    assert Z>=2
    if D is None:
        D = Z-1
    assert D>=Z-1
    I = np.eye(Z-1)
    l = np.ones((1,Z-1))* ((1 - np.sqrt(Z)) / (Z-1))
    ret = np.vstack((I,l))

    c = np.mean(ret,axis=0)
    ret = ret-c

    norms = np.linalg.norm(ret, axis=1, keepdims=True)
    norms[norms == 0] = 1
    ret = ret / norms

    vec = np.zeros((Z,D))
    vec[:,0:Z-1] = ret
    Q = generate_random_rotation_matrix(D)
    ret = np.matmul(vec, Q)
    return ret

def compute_pairwise_distances(vectors):
    num_vectors = vectors.shape[0]
    distances = np.zeros((num_vectors, num_vectors))

    for i in range(num_vectors):
        for j in range(i + 1, num_vectors):
            distance = np.linalg.norm(vectors[i] - vectors[j])
            distances[i, j] = distance
            distances[j, i] = distance  # symmetric matrix

    return distances

if __name__ == '__main__':
    v = generate_rand_regular_simplex_with_Z_vertices(4,10)
    print(v)
    print(compute_pairwise_distances(vectors=v))