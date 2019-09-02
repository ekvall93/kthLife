from utils import *
import gensim
from sklearn.mixture import BayesianGaussianMixture


model = gensim.models.Word2Vec.load("assets/finalproduct/lifeScienceDocvec/lifeScienceDocvec")
doc_vec = model.docvecs.vectors_docs
doc_tag = list(model.docvecs.doctags.keys())
word_vectors = model.wv.vectors

pickle_o = pickle_obj(); 
id_to_auth = pickle_o.load("assets/dictionaries/id_to_all_auths_2004")
auth_to_id = pickle_o.load("assets/dictionaries/auths_to_all_id_2004")
id_to_auth = pickle_o.load("assets/dictionaries/id_to_all_auths_2004")
auth_to_id = pickle_o.load("assets/dictionaries/auths_to_all_id_2004")



df_auth = pd.read_csv("assets/dataframes/KT_auth_2004")
df_abs = pd.read_csv("assets/dataframes/all_authors_df_2004")



#model = gensim.models.Word2Vec.load("assets/doc2vecModels/KTH2004_i10000_w10_d500_plainTrain/KTH2004_i10000_w10_d500_plainTrain.model")
#model = gensim.models.Word2Vec.load("assets/doc2vecModels/KTH2004_i2000_w10_d500_plainTrain_small/KTH2005_i2000_w10_d500_plainTrain")
model = gensim.models.Word2Vec.load("assets/doc2vecModels/KTH2004_i10000_w10_d500_plainTrain/KTH2004_i10000_w10_d500_plainTrain.model")


doc_vec = model.docvecs.vectors_docs
doc_tag = list(model.docvecs.doctags.keys())
word_vectors = model.wv.vectors

pickle_o = pickle_obj(); 
id_to_auth = pickle_o.load("assets/dictionaries/id_to_all_auths_2004")
auth_to_id = pickle_o.load("assets/dictionaries/auths_to_all_id_2004")
id_to_auth = pickle_o.load("assets/dictionaries/id_to_all_auths_2004")
auth_to_id = pickle_o.load("assets/dictionaries/auths_to_all_id_2004")

#negative_test = pickle_o.load("assets/goldenstandards/test_neg_aricles")
#negative_pos = pickle_o.load("assets/goldenstandards/test_pos_aricles")

negative_val = np.load("assets/goldenstandards/NLS_val.npy")
negative_test = np.load("assets/goldenstandards/NLS_test.npy")
postive_val = np.load("assets/goldenstandards/LS_val.npy")
positve_test = np.load("assets/goldenstandards/LS_test.npy")


tes_tag_str = list(np.array(list(negative_val) + list(negative_test)+list(postive_val)+list(positve_test)).astype(str))
train_tag = list(set(doc_tag) - (set(tes_tag_str)))
train_vec = model[train_tag]

test_nls = list(np.asarray(negative_test).astype(str))
val_nls = list(np.asarray(negative_val).astype(str))
test_ls = list(np.asarray(positve_test).astype(str))
val_ls = list(np.asarray(postive_val).astype(str))

X_nls_test = model[test_nls]
X_nls_val = model[val_nls]

X_ls_test = model[test_ls]
X_ls_val = model[val_ls]


y_nls_test = np.ones(X_nls_test.shape[0])
y_nls_val = np.ones(X_nls_val.shape[0])

y_ls_test = np.zeros(X_ls_test.shape[0])
y_ls_val = np.zeros(X_ls_val.shape[0])


X_test = np.concatenate((X_nls_test, X_ls_test))
X_val = np.concatenate((X_nls_val, X_ls_val))

y_test = np.concatenate((y_nls_test, y_ls_test))
y_val = np.concatenate((y_nls_val, y_ls_val))

all_vec = np.concatenate([train_vec, word_vectors])
tags =np.concatenate([train_tag, model.wv.index2word])
all_vec = np.asarray(all_vec)
tags =np.asarray(tags)


BGMM = BayesianGaussianMixture(1500, verbose=2, max_iter=150)

labels = BGMM.fit_predict(all_vec)

np.save("assets/clusterModels/BGMM_lables1500_removedTestVal_docvec10_000",labels)
save_classifer(BGMM, "assets/clusterModels/BGMM1500_removedTestVal_docvec10_000")


