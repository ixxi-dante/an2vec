"""Process the output of data-to-csv.sh to produce an npz file usable
by an2vec.

This script gets the mutual mention largest connected component of a
weighted edgelist, computes Word2Vec vectors, clusters them,
and saves the adjacency matrix and user features to a format
usable by an2vec.

"""

import os
import time
import pickle
import logging
import itertools
from operator import itemgetter
from collections import defaultdict
from argparse import ArgumentParser

import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix
from gensim.models import Word2Vec
import gensim
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.cluster import SpectralClustering


UINT32_MAX_VALUE = np.iinfo(np.uint32).max
SIMILARITIES_THRESHOLD = 1e-9

logging.basicConfig()
logging.root.setLevel(logging.NOTSET)


def parse_args():
    """Parse all script arguments from the command line return
    a struct containing them."""

    parser = ArgumentParser(description="Process the output of data-to-csv.sh "
                            "to get the mutual mention largest connected "
                            "component, compute Word2Vec vectors, cluster "
                            "them, and save the adjacency matrix and user "
                            "features to a format usable by an2vec.")

    parser.add_argument('--mutual_threshold', type=int, required=True,
                        help="Number of mutual mentions at or above which "
                        "we create an edge (e.g. 5)")
    parser.add_argument('--tweet_min_words', type=int, required=True,
                        help="Minimal number of words for a tweet to be "
                        "included in Word2Vec (e.g. 3)")
    parser.add_argument('--w2v_dim', type=int, required=True,
                        help="Dimension of word2vec vectors (e.g. 50)")
    parser.add_argument('--w2v_iter', type=int, required=True,
                        help="Number of epochs for which word2vec should "
                        "go over the data (e.g. 10)")
    parser.add_argument('--cluster_hashtags_only', type=bool, required=True,
                        help="Whether to cluster only hashtags or all words")
    parser.add_argument('--nclusters', type=int, required=True,
                        help="Number of clusters to compute for word vectors "
                        "(e.g. 100)")

    parser.add_argument('--nworkers', type=int, required=True,
                        help="Number of threads for word2vec, similarities "
                        "computation, and spectral clustering; use -1 for "
                        "'all cpus'")

    parser.add_argument('--weighted_edgelist_path', type=str, required=True,
                        help="Path to the weighted edgelist produced by "
                        "data-to-csv.sh")
    parser.add_argument('--user_tweets_path', type=str, required=True,
                        help="Path to the user tweets csv file produced by "
                        "data-to-csv.sh")

    parser.add_argument('--npz_path', type=str, required=True,
                        help="Path to save the dataset npz file usable by "
                        "an2vec")

    parser.add_argument('--cluster2words_path', type=str,
                        help="Path to save the pickled assocation of a "
                        "cluster to its words; if not provided, "
                        "skip saving")
    parser.add_argument('--uid2orig_path', type=str,
                        help="Path to save the pickled map of user ids to "
                        "ids in the full graph (extracted from "
                        "weighted_edgelist_path); if not provided, skip "
                        "saving")
    parser.add_argument('--orig2lcc_path', type=str,
                        help="Path to save the pickled map of indices in the "
                        "full graph to indices in the mutual mention largest "
                        "connected component; if not provided, skip saving")
    args = parser.parse_args()

    if args.cluster2words_path is None:
        logging.warning("Will not save cluster2words")
    if args.uid2orig_path is None:
        logging.warning("Will not save uid2orig")
    if args.orig2lcc_path is None:
        logging.warning("Will not save orig2lcc")

    return args


def is_hashtag(word):
    """Test if a word is a hashtag."""
    return len(word) > 1 and word.startswith('#')


def is_edge_mutual(graph, edge, threshold):
    """Test if `edge` in `graph` has wight >= `threshold` in both
    directions."""
    src, dst = edge
    try:
        min_weight = min(graph.edges[src, dst]['weight'],
                         graph.edges[dst, src]['weight'])
        return min_weight >= threshold
    except KeyError:
        return False


def get_stopwords():
    """Get the set of stopwords."""
    with open('data/stopwords.txt', 'r') as file:
        return set(file.read().split())


def check_uint32_encodable(array):
    """Is `array` encodable as np.uint32."""
    assert array.min() >= 0, "Can't encode as np.uint32"
    assert array.max() <= UINT32_MAX_VALUE, "Can't encode as np.uint32"


def csv_to_adj(filepath):
    """Read a `weight,src,dst` csv file into a COO adjacency matrix.

    Return the adjacency matrix in COO format, and the map of node ids in
    the file to ids in the matrix.

    """

    # user_id -> matrix_id
    uid2adj = dict()
    nids = 0

    # Prepare coo_matrix components
    rows = []
    cols = []
    data = []

    with open(filepath, "r") as csvfile:
        for line in csvfile:
            try:
                weight, src, dst = [int(v) for v in line.split(',')]
            except ValueError:
                # Some lines have `null` user ids, thanks to
                # the original SoSweet data
                pass
            if src not in uid2adj.keys():
                uid2adj[src] = nids
                nids += 1
            if dst not in uid2adj.keys():
                uid2adj[dst] = nids
                nids += 1
            rows.append(uid2adj[src])
            cols.append(uid2adj[dst])
            data.append(weight)

    # Build the adjacency matrix
    rows = np.array(rows)
    cols = np.array(cols)
    data = np.array(data)
    check_uint32_encodable(data)
    adj = coo_matrix((data, (rows, cols)), shape=(nids, nids), dtype=np.uint32)
    adj.sum_duplicates()

    return adj, uid2adj


def build_adj_lcc(args):
    """Build the mutual mention largest connected component from
    a csv edgelist with weights."""

    logging.info("Read '%s' into an adjacency matrix",
                 args.weighted_edgelist_path)
    adj_orig, uid2orig = csv_to_adj(args.weighted_edgelist_path)

    logging.info("Convert adjacency matrix to a graph")
    g_orig = nx.from_scipy_sparse_matrix(adj_orig, create_using=nx.DiGraph,
                                         edge_attribute='weight')

    logging.info("Extract mutual mention graph from full graph")
    g_mutual = nx.edge_subgraph(
        g_orig,
        [e for e in g_orig.edges
         if is_edge_mutual(g_orig, e, args.mutual_threshold)]
    )
    g_mutual = nx.Graph(g_mutual)

    logging.info("Extract largest connected component from "
                 "mutual mention graph")
    g_lcc = g_mutual.subgraph(max(nx.connected_components(g_mutual), key=len))

    logging.info("Relabel largest connected component node ids")
    orig2lcc = dict((id, n) for n, id in enumerate(g_lcc.nodes))
    g_lcc = nx.relabel_nodes(g_lcc, orig2lcc)
    assert set(g_lcc.nodes) == set(range(len(g_lcc.nodes))), \
        "Mutual mention lcc does not cover all node ids"

    logging.info("Convert largest connected component to adjacency matrix")
    adj_lcc = nx.to_scipy_sparse_matrix(g_lcc, dtype=np.uint32,
                                        weight='weight',
                                        nodelist=sorted(g_lcc.nodes()))
    adj_lcc = coo_matrix(adj_lcc)

    logging.info("Number of nodes in the largest connected component = %s",
                 len(g_lcc.nodes))

    return adj_lcc, uid2orig, orig2lcc


def build_uid2tweets(args, uid2orig, orig2lcc):
    """Build the association of user ids to user tweets (split on words,
    filtered for useless words)."""

    # Prepare output variables
    uid2tweets = defaultdict(list)
    nextid = len(uid2orig)

    logging.info("Read '%s' into an ossociation of user ids to tweets",
                 args.user_tweets_path)
    stopwords = get_stopwords()
    with open(args.user_tweets_path, "r") as tweetsfile:
        for line in tweetsfile:
            # Extract user id and tweet from the line
            parts = line.split()
            uid, tweet = int(parts[0]), parts[1:]

            # Keep only words that interest us:
            # hashtags, only alphabetic characters, no stopwords
            tweet_clean = []
            for word in tweet:
                if (word[0] == '#'
                        or (word.isalpha() and word not in stopwords)):
                    tweet_clean.append(word)

            # Tweets with less than a few words don't contain any information
            if len(tweet_clean) < args.tweet_min_words:
                continue

            # Some users are not in the full graph
            # so we add their user id to our list
            if uid not in uid2orig:
                uid2orig[uid] = nextid
                nextid += 1

            # Only store tweets for users in
            # the mutual mention largest connected component
            if uid2orig[uid] in orig2lcc:
                uid2tweets[uid].append(tweet_clean)

    return uid2tweets


def build_w2v_model(args, uid2tweets):
    """Build Word2Vec model using provided list of tweets."""

    all_tweets = list(itertools.chain.from_iterable(uid2tweets.values()))
    logging.info("Build word2vec model over %s tweets", len(all_tweets))

    # gensim's word2vec doesn't seem to take `-1` as meaning 'use all cpus'
    nworkers = args.nworkers
    if nworkers == -1:
        nworkers = os.cpu_count()

    w2v_start = time.time()
    w2v_model = Word2Vec(all_tweets,
                         size=args.w2v_dim,
                         iter=args.w2v_iter,
                         workers=nworkers)
    w2v_end = time.time()
    logging.info("Built word2vec model in %s seconds", w2v_end - w2v_start)

    return w2v_model


def cluster_word_vectors(args, w2v_model):
    """Cluster word vectors save the association of cluster to words,
    and return the association of word to cluster."""

    vocabulary = list(w2v_model.wv.vocab.keys())
    if args.cluster_hashtags_only:
        _w_ht = 'hashtag'
        vocabulary = list(filter(is_hashtag, vocabulary))
    else:
        _w_ht = 'word'
    logging.info("Cluster Word2Vec's %s %ss into %s clusters",
                 len(vocabulary), _w_ht, args.nclusters)

    logging.info("Get normalised %s vectors", _w_ht)
    vectors = w2v_model.wv[vocabulary]
    vectors_normalised = np.array([gensim.matutils.unitvec(vec)
                                   for vec in vectors])
    logging.info("Pre-compute %s similarities", _w_ht)
    similarities = pairwise_kernels(vectors_normalised, metric="cosine",
                                    n_jobs=args.nworkers)
    # cosine goes from -1 to 1, when we want similarities going from 0 to 1
    # (and not 0 as that crashes the clustering algorithm)
    similarities[similarities < SIMILARITIES_THRESHOLD] = \
        SIMILARITIES_THRESHOLD

    logging.info("Spectral-cluster based on %s similarities", _w_ht)
    spectral = SpectralClustering(n_clusters=args.nclusters,
                                  affinity="precomputed",
                                  n_jobs=args.nworkers)
    spectral.fit(similarities)
    logging.info("Get each %s's cluster and each cluster's %ss", _w_ht, _w_ht)
    cluster2words = defaultdict(list)
    word2cluster = dict()
    for i, cluster in enumerate(spectral.fit_predict(similarities)):
        cluster2words[cluster].append(vocabulary[i])
        word2cluster[vocabulary[i]] = cluster

    if args.cluster2words_path is None:
        logging.info("Not saving cluster2words")
    else:
        logging.info("Save cluster2words to '%s'", args.cluster2words_path)
        with open(args.cluster2words_path, 'wb') as cluster2words_file:
            pickle.dump(cluster2words, cluster2words_file)

    return word2cluster


def build_user_features_lcc(uid2orig, orig2lcc, uid2tweets, word2cluster):
    """Build the COO matrix of user features, for users in
    the mutual mention largest connected component."""

    logging.info("Build user feature matrix from word/hashtag clustering")
    rows = []
    cols = []
    data = []
    for uid, tweets in uid2tweets.items():
        for tweet in tweets:
            for word in tweet:
                if word not in word2cluster:
                    continue
                rows.append(uid2orig[uid])
                cols.append(word2cluster[word])
                data.append(1)

    rows = np.array(rows)
    cols = np.array(cols)
    data = np.array(data)
    check_uint32_encodable(data)
    user_features = coo_matrix((data, (rows, cols)), dtype=np.uint32)
    user_features.sum_duplicates()

    logging.info("Keep only users present in the mutual "
                 "mention largest connected component")
    lcc2orig_list = list(map(itemgetter(0),
                             sorted(orig2lcc.items(), key=itemgetter(1))))
    user_features_lcc = user_features\
        .toarray()[lcc2orig_list, :]\
        .astype(np.float)

    logging.info("l1-normalise user features")
    nonzero_users = np.where(user_features_lcc.sum(1))[0]
    user_features_lcc[nonzero_users] = (
        user_features_lcc[nonzero_users]
        / user_features_lcc[nonzero_users].sum(1, keepdims=True)
    )

    logging.info("%s users with all-zero features (none of their "
                 "words are in our clustering)",
                 user_features_lcc.shape[0] - len(nonzero_users))
    return coo_matrix(user_features_lcc)


def main():
    """Read csv edgelist and tweets, get the mutual mention largest
    connected component, compute Word2Vec vectors, cluster words or hashtags,
    and save the adjacency matrix and user features to a format
    usable by an2vec."""

    args = parse_args()
    adj_lcc, uid2orig, orig2lcc = build_adj_lcc(args)
    uid2tweets = build_uid2tweets(args, uid2orig, orig2lcc)
    w2v_model = build_w2v_model(args, uid2tweets)
    word2cluster = cluster_word_vectors(args, w2v_model)
    user_features_lcc = build_user_features_lcc(uid2orig, orig2lcc,
                                                uid2tweets, word2cluster)

    logging.info("Save adjacency and features (and zeroed labels) to '%s'",
                 args.npz_path)
    np.savez_compressed(args.npz_path,
                        adjdata=np.ones_like(adj_lcc.data),
                        adjcol=adj_lcc.col,
                        adjrow=adj_lcc.row,
                        features=user_features_lcc.toarray(),
                        labels=np.zeros((adj_lcc.shape[0], 1)))

    if args.uid2orig_path is None:
        logging.info("Not saving uid2orig")
    else:
        logging.info("Saving uid2orig to '%s'", args.uid2orig_path)
        with open(args.uid2orig_path, 'wb') as file:
            pickle.dump(uid2orig, file)
    if args.orig2lcc_path is None:
        logging.info("Not saving orig2lcc")
    else:
        logging.info("Saving orig2lcc to '%s'", args.orig2lcc_path)
        with open(args.orig2lcc_path, 'wb') as file:
            pickle.dump(orig2lcc, file)


if __name__ == '__main__':
    main()
