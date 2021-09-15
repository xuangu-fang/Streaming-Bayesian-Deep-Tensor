import math

import theano

import theano.tensor as T

class Embedding:

    def __init__(self, m_w_init, v_w_init):

        self.m_u = theano.shared(value = m_w_init.astype(theano.config.floatX),
            name='m_u', borrow = True)

        self.v_u = theano.shared(value = v_w_init.astype(theano.config.floatX),
            name='v_u', borrow = True)
