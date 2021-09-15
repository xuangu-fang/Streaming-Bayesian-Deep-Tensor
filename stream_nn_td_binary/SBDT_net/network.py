
import numpy as np

import theano

import theano.tensor as T

import network_layer

import embedding

class Network:

    #zsd: define a vector representing the whole
    def __init__(self, m_w_init, v_w_init, m_u_init, v_u_init, a_init, b_init,n_stream_batch):
        # We create the different layers
        self.n_stream_batch = n_stream_batch
        self.layers = []

        if len(m_w_init) > 1:
            for m_w, v_w in zip(m_w_init[ : -1 ], v_w_init[ : -1 ]):
                self.layers.append(network_layer.Network_layer(m_w, v_w, True))

        self.layers.append(network_layer.Network_layer(m_w_init[-1], v_w_init[-1], False))
        #self.layers.append(network_layer.Network_layer(m_w_init[-1],v_w_init[-1], True))

        # We create mean and variance parameters from all layers

        self.params_m_w = []
        self.params_v_w = []
        #self.params_w = []
        for layer in self.layers:
            self.params_m_w.append(layer.m_w)
            self.params_v_w.append(layer.v_w)
            #self.params_w.append(layer.m_w)

        # We create mean and variance parameters from all embedding elements

        self.params_embed = []

        if len(m_u_init) > 1:
            for m_u, v_u in zip(m_u_init, v_u_init):
                self.params_embed.append(embedding.Embedding(m_u, v_u))

        
        self.params_m_u = []
        self.params_v_u = []

        for embed in self.params_embed: # mode
            self.params_m_u.append(embed.m_u)
            self.params_v_u.append(embed.v_u)

        # We create the theano variables for a and b

        self.a = theano.shared(float(a_init))
        self.b = theano.shared(float(b_init))
        #zsd
        #self.b = theano.shared(float(b_init)/self.n_stream_batch)
        
        # self.x = theano.shared((np.ones(3)))

       
    def output_deterministic(self, x):

        # Recursively compute output
        #self.x = self.x.set_value(x.get_value())
        x = self.get_embed(x)

        for layer in self.layers:
            x = layer.output_deterministic(x)

        return x[0]

    def output_probabilistic(self, m):

        v = T.zeros_like(m)

        # Recursively compute output

        for layer in self.layers:
            m, v = layer.output_probabilistic(m, v)

        return (m[ 0 ], v[ 0 ])

    #zsd: calc. everything here
    def logZ_Z1_Z2(self, x, y):
        tau = self.a/self.b
        v = 0.0
        f = self.output_deterministic(x[0]) 
        y = y[0]
        #w
        for i in range(len(self.params_m_w)):
            prod = T.grad(f, self.params_m_w[i])**2*self.params_v_w[i]
            v = v + prod.sum()
        #u
        for i in range(len(self.params_m_u)):#mode
            prod = T.grad(f, self.params_m_u[i])**2*self.params_v_u[i]
            v = v + prod.sum()


        cdf_input = (2*y-1)*f/T.sqrt(1+v)

        logZ = T.log(0.5*(1 + T.erf(cdf_input/1.4142135)))


        # v_final = v + 1.0/tau
        # logZ = -0.5 * (T.log(2*3.1415926*v_final) + (y - f)**2 / v_final)

        # not use            
        a_star = self.a + 0.5
        b_star = self.b + 0.5*((y-f)**2 + v)

        '''
        m, v = self.output_probabilistic(x)

        v_final = v + self.b / (self.a - 1)
        v_final1 = v + self.b / self.a
        v_final2 = v + self.b / (self.a + 1)

        logZ = -0.5 * (T.log(v_final) + (y - m)**2 / v_final)
        logZ1 = -0.5 * (T.log(v_final1) + (y - m)**2 / v_final1)
        logZ2 = -0.5 * (T.log(v_final2) + (y - m)**2 / v_final2)
        '''

        return (logZ, a_star, b_star)

    #update the parameters for all weights and the lam in eq(11)
    def generate_updates(self,x, logZ, a_star, b_star):
        updates = []
        for i in range(len(self.params_m_w)):
            updates.append((self.params_m_w[ i ], self.params_m_w[ i ] + \
                self.params_v_w[ i ] * T.grad(logZ, self.params_m_w[ i ])))

            updates.append((self.params_v_w[ i ], self.params_v_w[ i ] - \
               self.params_v_w[ i ]**2 * \
                (T.grad(logZ, self.params_m_w[ i ])**2 - 2 * \
                T.grad(logZ, self.params_v_w[ i ]))))


        for i in range(len(self.params_m_u)):#mod

            # indx =  x[i]

            updates.append((self.params_m_u[ i ], self.params_m_u[ i ] + \
                self.params_v_u[ i ]* T.grad(logZ, self.params_m_u[ i ])))   

            updates.append((self.params_v_u[ i ], self.params_v_u[ i ] - \
               self.params_v_u[ i ]**2 * \
                (T.grad(logZ, self.params_m_u[ i ])**2 - 2 * \
                T.grad(logZ, self.params_v_u[ i ]))))         
        
        '''
        updates.append((self.a, self.a + \
                self.b * T.grad(logZ, self.a)))
        updates.append((self.b, self.b - \
               self.b**2 * \
                (T.grad(logZ, self.a)**2 - 2 * \
                0.0)))
        '''

        #zsd: CEP update
        # updates.append((self.a, a_star))
        # updates.append((self.b, b_star))
        
        '''
        updates.append((self.a, 1.0 / (T.exp(logZ2 - 2 * logZ1 + logZ) * \
            (self.a + 1) / self.a - 1.0)))
        updates.append((self.b, 1.0 / (T.exp(logZ2 - logZ1) * (self.a + 1) / \
            (self.b) - T.exp(logZ1 - logZ) * self.a / self.b)))
        '''

        return updates

    def get_embed(self,x):

        embed_list = []
        # for i in range(len(self.params_embed)):
        for i in range(len(self.params_m_u)):
            indx = x[i]
            embed_list.append(self.params_m_u[i].take(indx,0))
        return T.concatenate(embed_list).flatten()

    def get_index(self,x):

            return x
           

    def get_params(self):

        m_w = []
        v_w = []
        for layer in self.layers:
            m_w.append(layer.m_w.get_value())
            v_w.append(layer.v_w.get_value())

        m_u = []
        v_u = []
        for embed in self.params_embed:
            m_u.append(embed.m_u.get_value())
            v_u.append(embed.v_u.get_value())

        

        return { 'm_w': m_w, 'v_w': v_w ,'m_u': m_u, 'v_u': v_u , 'a': self.a.get_value(),
            'b': self.b.get_value() }

    def set_params(self, params):

        for i in range(len(self.layers)):
            self.layers[ i ].m_w.set_value(params[ 'm_w' ][ i ])
            self.layers[ i ].v_w.set_value(params[ 'v_w' ][ i ])

        for i in range(len(self.params_embed)):
            self.params_embed[ i ].m_u.set_value(params[ 'm_u' ][ i ])
            self.params_embed[ i ].v_u.set_value(params[ 'v_u' ][ i ])       


        self.a.set_value(params[ 'a' ])
        self.b.set_value(params[ 'b' ])

    def remove_invalid_updates(self, new_params, old_params):

        m_w_new = new_params[ 'm_w' ]
        v_w_new = new_params[ 'v_w' ]
        m_w_old = old_params[ 'm_w' ]
        v_w_old = old_params[ 'v_w' ]

        a_old = old_params[ 'a' ]
        a_new = new_params[ 'a' ]
        b_old = old_params[ 'b' ]
        b_new = new_params[ 'b' ]

        m_u_new = new_params[ 'm_u' ]
        v_u_new = new_params[ 'v_u' ]
        m_u_old = old_params[ 'm_u' ]
        v_u_old = old_params[ 'v_u' ]

        for i in range(len(self.layers)):
            index1 = np.where(v_w_new[ i ] <= 1e-100)
            index2 = np.where(np.logical_or(np.isnan(m_w_new[ i ]),
                np.isnan(v_w_new[ i ])))

            index = [ np.concatenate((index1[ 0 ], index2[ 0 ])),
                np.concatenate((index1[ 1 ], index2[ 1 ])) ]

            if len(index[ 0 ]) > 0:
                m_w_new[ i ][ tuple(index) ] = m_w_old[ i ][ tuple(index) ]
                v_w_new[ i ][ tuple(index) ] = v_w_old[ i ][ tuple(index) ]

        if np.isnan(a_new) or np.isnan(b_new) or b_new <= 1e-100:
            new_params[ 'a' ] = a_old
            new_params[ 'b' ] = b_old


        for i in range(len(self.params_embed)):
            index1 = np.where(v_u_new[ i ] <= 1e-100)
            index2 = np.where(np.logical_or(np.isnan(m_u_new[ i ]),
                np.isnan(v_u_new[ i ])))

            index = [ np.concatenate((index1[ 0 ], index2[ 0 ])),
                np.concatenate((index1[ 1 ], index2[ 1 ])) ]

            if len(index[ 0 ]) > 0:
                m_u_new[ i ][ tuple(index) ] = m_u_old[ i ][ tuple(index) ]
                v_u_new[ i ][ tuple(index) ] = v_u_old[ i ][ tuple(index) ]
        


        

    '''
    def sample_w(self):

        w = []
        for i in range(len(self.layers)):
            w.append(self.params_m_w[ i ].get_value() + \
                np.random.randn(self.params_m_w[ i ].get_value().shape[ 0 ], \
                self.params_m_w[ i ].get_value().shape[ 1 ]) * \
                np.sqrt(self.params_v_w[ i ].get_value()))

        for i in range(len(self.layers)):
            self.params_w[ i ].set_value(w[ i ])
    '''
