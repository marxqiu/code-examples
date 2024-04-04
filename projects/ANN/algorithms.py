import SPTAG
import hnswlib
import numpy as np
import faiss
import sklearn.preprocessing
import pyflann



class FLANN():
    def __init__(self, metric, target_precision):
        self._target_precision = target_precision
        self.name = 'FLANN(target_precision=%f)' % self._target_precision
        self._metric = metric

    def fit(self, X):
        if X.dtype != np.float32:
            X = X.astype(np.float32)
            
        self._flann = pyflann.FLANN(
            target_precision=self._target_precision,
            algorithm='autotuned', log_level='info')
        
        if self._metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')
        self._flann.build_index(X)

    def query(self, v, n):
        if self._metric == 'angular':
            v = sklearn.preprocessing.normalize([v], axis=1, norm='l2')[0]
        if v.dtype != np.float32:
            v = v.astype(np.float32)
        return self._flann.nn_index(v, n)[0][0]

class spann():
    def __init__(self, mode, metric):
        self._mode = mode
        self._metric = {'angular': 'Cosine', 'euclidean': 'L2'}[metric]
        
    def fit(self, X):
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        
        self._index = SPTAG.AnnIndex(self._mode, 'Float', X.shape[1])
        self._index.SetBuildParam("NumberOfThreads", '32', "Index")
        self._index.SetBuildParam("DistCalcMethod", self._metric, "Index") 
        self._index.Build(X, X.shape[0], False)


    def set_query_arguments(self, MaxCheck):
        self._maxCheck = MaxCheck
        self._index.SetSearchParam("MaxCheck", str(MaxCheck), "Index")

    def query(self, v, k):
        # nearest K ids
        if v.dtype != np.float32:
            v = v.astype(np.float32)
        result = self._index.Search(v, k)
        #return [(id, dist) for id, dist in zip(result[0], result[1])]
        return result[0]#only returns the idices

    def __str__(self):
        return 'Sptag(check=%d)' % self._maxCheck
    
    
class hnsw():
    def __init__(self, ef_construction, M, metric):
        self.metric = {'angular': 'cosine', 'euclidean': 'l2'}[metric]
        self.ef_construction = ef_construction
        self.M = M
        
    
    def fit(self, X):
    # Declaring index
        self.p = hnswlib.Index(space = self.metric, dim = X.shape[1]) # possible options are l2, cosine or ip

    # Initializing index - the maximum number of elements should be known beforehand
        self.p.init_index(max_elements = len(X), ef_construction = self.ef_construction, M = self.M)

    # Element insertion (can be called several times):
        self.p.add_items(X, np.arange(len(X)))
    
    def set_query_arguments(self, ef):
        self.ef = ef
    # Controlling the recall by setting ef:
        self.p.set_ef(self.ef) # ef should always be > k

    # Query dataset, k - number of closest elements (returns 2 numpy arrays)
    def query(self, v, k):
        labels, distances = self.p.knn_query(np.expand_dims(v, axis=0), k)
        #return [(id, dist) for id, dist in zip(labels[0], distances[0])]
        return labels[0]
    
    def __str__(self):
        return 'hnsw(ef_construction=%d, M=%d, ef=%d)' % (self.ef_construction, self.M, self.ef)

    

class IVFADCR():
    def __init__(self, n_lists, M, nbits, M_refine, nbits_refine, metric):
        self._n_list = n_lists#number of centroids
        self._metric = metric
        self._M = M#number of subquantizer, power of two 
        self._nbits = nbits#sqrt(len(X)) * C
        self._M_refine = M_refine
        self._nbits_refine = nbits_refine
        
    def fit(self, X):
        if X.dtype != np.float32:
            X = X.astype(np.float32)
            
        if self._metric == 'angular':
            faiss.normalize_L2(X)
        
        
        
        self.quantizer = faiss.IndexFlatL2(X.shape[1])
        index = faiss.IndexIVFPQR(
            self.quantizer, X.shape[1], self._n_list, self._M, self._nbits, self._M_refine, self._nbits_refine) #Here we compress d 32-bit floats to M bytes, so the compression factor is 32d / 8M = 4d / M.

        index.metric_type = faiss.METRIC_INNER_PRODUCT if self._metric == 'angular' else faiss.METRIC_L2

        index.train(X)
        index.add(X)
        self.index = index
        
    def query(self, v, k):
        if self._metric == 'angular':
            v /= np.linalg.norm(v)
        D, I = self.index.search(np.expand_dims(
            v, axis=0).astype(np.float32), k)
        return I[0]

    def set_query_arguments(self, n_probe):
        faiss.cvar.indexIVF_stats.reset()
        self._n_probe = n_probe
        self.index.nprobe = self._n_probe

    def get_additional(self):
        return {"dist_comps": faiss.cvar.indexIVF_stats.ndis +      # noqa
                faiss.cvar.indexIVF_stats.nq * self._n_list}

    def __str__(self):
        return 'IVFADCR(n_list=%d, n_probe=%d, M=%d, nbits=%d, M_refine=%d, nbits_refine=%d)' % (self._n_list, 
                                                                                                 self._n_probe, 
                                                                                                 self._M, 
                                                                                                 self._nbits, #sqrt(len(X)) * C
                                                                                                 self._M_refine,
                                                                                                 self._nbits_refine )
    
    
    
class spann_ssd():
    def __init__(self, mode, metric, dataset):
        self._mode = mode
        self._metric = {'angular': 'Cosine', 'euclidean': 'L2'}[metric]
        self._dataset = dataset
        
    def fit(self, X):
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        
        self._index = SPTAG.AnnIndex('SPANN', 'Float', X.shape[1])
        
        self._index.SetBuildParam("IndexAlgoType", self._mode, "Base")
        self._index.SetBuildParam("DistCalcMethod", self._metric, "Base") 
        self._index.SetBuildParam("IndexDirectory", self._dataset + self._mode, "Base")
        
        
        
        
        self._index.SetBuildParam("isExecute", "true", "SelectHead")
        self._index.SetBuildParam("TreeNumber", "1", "SelectHead")
        self._index.SetBuildParam("BKTKmeansK", "32", "SelectHead")
        self._index.SetBuildParam("BKTLeafSize", "8", "SelectHead")
        self._index.SetBuildParam("SamplesNumber", "1000", "SelectHead")
        self._index.SetBuildParam("SaveBKT", "false", "SelectHead")
        self._index.SetBuildParam("SelectThreshold", "50", "SelectHead")
        self._index.SetBuildParam("SplitFactor", "6", "SelectHead")
        self._index.SetBuildParam("SplitThreshold", "100", "SelectHead")
        self._index.SetBuildParam("Ratio", "0.16", "SelectHead")
        self._index.SetBuildParam("NumberOfThreads", "64", "SelectHead")
        self._index.SetBuildParam("BKTLambdaFactor", "-1", "SelectHead")

        self._index.SetBuildParam("isExecute", "true", "BuildHead")
        self._index.SetBuildParam("NeighborhoodSize", "32", "BuildHead")
        self._index.SetBuildParam("TPTNumber", "32", "BuildHead")
        self._index.SetBuildParam("TPTLeafSize", "2000", "BuildHead")
        self._index.SetBuildParam("MaxCheck", "8192", "BuildHead")
        self._index.SetBuildParam("MaxCheckForRefineGraph", "8192", "BuildHead")
        self._index.SetBuildParam("RefineIterations", "3", "BuildHead")
        self._index.SetBuildParam("NumberOfThreads", "64", "BuildHead")
        self._index.SetBuildParam("BKTLambdaFactor", "-1", "BuildHead")

        self._index.SetBuildParam("isExecute", "true", "BuildSSDIndex")
        self._index.SetBuildParam("BuildSsdIndex", "true", "BuildSSDIndex")
        self._index.SetBuildParam("InternalResultNum", "64", "BuildSSDIndex")
        self._index.SetBuildParam("ReplicaCount", "8", "BuildSSDIndex")
        self._index.SetBuildParam("PostingPageLimit", "12", "BuildSSDIndex")
        self._index.SetBuildParam("NumberOfThreads", "64", "BuildSSDIndex")
        self._index.SetBuildParam("MaxCheck", "8192", "BuildSSDIndex")
        self._index.SetBuildParam("TmpDir", "/tmp/", "BuildSSDIndex")
        
        self._index.Build(X, X.shape[0], False)


    def set_query_arguments(self, MaxCheck, PostingLimit):
        self._maxCheck = MaxCheck
        self._postingLimit = PostingLimit
        self._index.SetSearchParam("isExecute", "true", "SearchSSDIndex")
        self._index.SetSearchParam("BuildSsdIndex", "false", "SearchSSDIndex")
        self._index.SetSearchParam("InternalResultNum", "32", "SearchSSDIndex")
        self._index.SetSearchParam("NumberOfThreads", "1", "SearchSSDIndex")
        self._index.SetSearchParam("HashTableExponent", "4", "SearchSSDIndex")
        self._index.SetSearchParam("ResultNum", "10", "SearchSSDIndex")
        self._index.SetSearchParam("MaxCheck", str(self._maxCheck), "SearchSSDIndex")#default 2048
        self._index.SetSearchParam("MaxDistRatio", "10000.0", "SearchSSDIndex")
        self._index.SetSearchParam("SearchPostingPageLimit", str(self._postingLimit), "SearchSSDIndex")#default 12


    def query(self, v, k):
        # nearest K ids
        if v.dtype != np.float32:
            v = v.astype(np.float32)
        result = self._index.Search(v, k)
        #return [(id, dist) for id, dist in zip(result[0], result[1])]
        return result[0]#only returns the idices

    def __str__(self):
        return 'Spann_ssd(check=%d)' % self._maxCheck