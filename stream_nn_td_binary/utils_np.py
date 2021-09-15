# numpy-based utility functions

import numpy as np
NP_FLOAT_TYPE = np.float64

# def get_concat_embeddings( U, inds):
#     nmod = len( U)
#     X = np.concatenate([U[i][inds[:, i]] for i in range(nmod)], axis=1)

#     assert  len( X.shape) == 2, 'Wrong Embedding shape'
#     return X

# def cal_f_rff( X, S, w ):
#     prj = X @ S.T
#     phi = np.concatenate([np.cos(prj), np.sin(prj)], axis=1)
#     f = np.reshape(phi @ w, -1)
#     return f
