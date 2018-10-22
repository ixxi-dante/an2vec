import numpy as np


def stbm(n_nodes=100, n_clusters=40,  # Network parameters
         n_topics=5, n_documents=30, n_slots=140, vocabulary_size=50,  # Language parameters
         ρ=None, π=None, θ=None):
    """Simulate a network + document scenario with STBM."""

    α = np.ones(n_topics)
    β = np.random.random(size=(n_topics, vocabulary_size))
    β /= np.sum(β, axis=1)[:, np.newaxis]
    if ρ is None:
        ρ = np.random.random(n_clusters)
        ρ /= np.sum(ρ)
    if π is None:
        pre_π = np.random.random((n_clusters, n_clusters))
        π = np.triu(pre_π, k=1) + np.triu(pre_π).T
    if θ is None:
        θ = np.random.dirichlet(alpha=α, size=(n_clusters, n_clusters))

    topics_range = range(n_topics)
    vocabulary_range = range(vocabulary_size)

    # Order Latent Variables
    Y = np.random.multinomial(1, pvals=ρ, size=n_nodes)
    A = np.random.binomial(1, Y @ π @ Y.T)
    Z = np.nan * np.zeros((n_nodes, n_nodes, n_documents, n_slots))
    W = np.nan * np.zeros((n_nodes, n_nodes, n_documents, n_slots))

    sources_A, destinations_A = np.where(A)
    for i, j in zip(sources_A, destinations_A):
        Z[i, j, :, :] = np.random.choice(topics_range,
                                         p=θ[Y[i].astype(bool), Y[j].astype(bool)][0],
                                         size=(n_documents, n_slots))

    for k in topics_range:
        k_idx = np.where(Z == k)
        W[k_idx] = np.random.choice(vocabulary_range, p=β[k], size=len(k_idx[0]))

    return (Y, A, Z, W)


def interpolate_θ_S3_S2(α, n_topics, n_clusters):
    # Verify arguments
    assert n_topics >= n_clusters  # in order to map one topic to each cluster
    assert α >= 0 and α <= 1  # for smooth interpolation from s3 to s2

    # Start from uniform theta
    θ = np.ones((n_clusters, n_clusters, n_topics)) / n_topics

    # Assign a random (distinct) topic to each cluster
    cluster2topic = np.arange(n_topics)
    np.random.shuffle(cluster2topic)
    cluster2topic = cluster2topic[:n_clusters]

    # Interpolate cluster-specificity
    for i in range(n_clusters):
        θ[i, i, :] = ((1 - α) / n_topics) * np.ones(n_topics)
        θ[i, i, cluster2topic[i]] = α + (1 - α) / n_topics

    return θ
