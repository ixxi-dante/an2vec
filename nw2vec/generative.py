import numba
import numpy as np

from nw2vec import utils


@numba.jit(nopython=True)
def stbm(n_nodes=100, n_clusters=40,  # Network parameters
         n_topics=5, n_documents=30, n_slots=140, vocabulary_size=50,  # Language parameters
         ρ=None, π=None, θ=None):
    """Simulate a network + document scenario with STBM."""

    β = np.random.random(size=(n_topics, vocabulary_size))
    β /= np.expand_dims(np.sum(β, axis=1), -1)
    if ρ is None:
        ρ = np.random.random(n_clusters)
        ρ /= np.sum(ρ)
    if π is None:
        pre_π = np.random.random((n_clusters, n_clusters))
        π = np.triu(pre_π, k=1) + np.triu(pre_π, k=0).T
    # θ cannot be None, but we can't enforce that with Numba, it'll just bail further down

    # Order Latent Variables
    Y = np.random.multinomial(1, pvals=ρ, size=n_nodes).astype(np.float_)
    A_probs = Y @ π @ Y.T
    A = np.zeros_like(A_probs)
    for i in numba.prange(len(A_probs.flat)):
        A.flat[i] = np.random.binomial(1, A_probs.flat[i])
    Z = np.nan * np.zeros((n_nodes, n_nodes, n_documents, n_slots))
    W = np.nan * np.zeros((n_nodes, n_nodes, n_documents, n_slots))

    sources_A, destinations_A = np.where(A)
    for i, j in zip(sources_A, destinations_A):
        qs, rs = np.where(Y[i])[0], np.where(Y[j])[0]
        assert len(qs) == len(rs) == 1
        q, r = qs[0], rs[0]
        topics_choice_aliases = utils.alias_setup(θ[q, r])
        for k in numba.prange(n_documents * n_slots):
            Z[i, j].flat[k] = utils.alias_draw(*topics_choice_aliases)

    for k in numba.prange(n_topics):
        k_idx = np.where(Z == k)
        vocabulary_choice_aliases = utils.alias_setup(β[k])
        for idx in zip(*k_idx):
            W[idx] = utils.alias_draw(*vocabulary_choice_aliases)

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
