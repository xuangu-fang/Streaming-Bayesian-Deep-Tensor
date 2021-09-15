
import numpy as np
np.random.seed(1)

import math
import theano

import theano.tensor as T

class Prior:

    def __init__(self, layer_sizes, var_targets, R, ndims):


        self.R = R
        self.ndims = ndims

        # We refine the factor for the prior variance on the weights

        n_samples = 3.0
        v_observed = 1.0
        self.a_w = 2.0 * n_samples
        self.b_w = 2.0 * n_samples * v_observed

        # hyperoarameter of s&s prior
        self.rho_0 = 0.5
        self.tau_0 = 1.0

        # We refine the factor for the prior variance on the embedding

        self.a_u = 2.0 * n_samples
        self.b_u = 2.0 * n_samples * v_observed

        # We refine the factor for the prior variance on the noise

        n_samples = 3.0
        a_sigma = 2.0 * n_samples
        b_sigma = 2.0 * n_samples * var_targets

        self.a_sigma_hat_nat = a_sigma - 1
        self.b_sigma_hat_nat = -b_sigma

        self.m_sigma = 0
        self.v_sigma = 1

        # We refine the Spike and slab prior on the weights

        self.rnd_m_w = []
        self.m_w_hat_nat = []
        self.v_w_hat_nat = []
        self.rho_w_hat_nat = []
        # self.a_w_hat_nat = []
        # self.b_w_hat_nat = []
        for size_out, size_in in zip(layer_sizes[ 1 : ], layer_sizes[ : -1 ]):
            self.rnd_m_w.append(1.0 / np.sqrt(size_in + 1) *
                np.random.randn(size_out, size_in + 1))
            self.m_w_hat_nat.append(np.zeros((size_out, size_in + 1)))
            self.v_w_hat_nat.append((self.a_w - 1) / self.b_w * \
                np.ones((size_out, size_in + 1)))
            self.rho_w_hat_nat.append(np.zeros((size_out, size_in + 1)))
            # self.a_w_hat_nat.append(np.zeros((size_out, size_in + 1)))
            # self.b_w_hat_nat.append(np.zeros((size_out, size_in + 1)))

        # We refine the gaussian prior on the embedding

        self.rnd_m_u = []
        self.m_u_hat_nat = []
        self.v_u_hat_nat = []
        self.a_u_hat_nat = []
        self.b_u_hat_nat = []

        nmod = len(ndims)
        for i in range(nmod): # each mode
            # rnd_m_u_mode = []
            # m_u_hat_nat_mode = []
            # v_u_hat_nat_mode = []
            # a_u_hat_nat_mode = []
            # b_u_hat_nat_mode = []
            # for j in range(ndims[i]): # dimension of each mode
            self.rnd_m_u.append(1/R * np.random.randn(ndims[i],R))
            self.m_u_hat_nat.append(np.zeros((ndims[i],R)))
            self.v_u_hat_nat.append((self.a_u - 1) / self.b_u * np.ones((ndims[i],R)))
            self.a_u_hat_nat.append(np.zeros((ndims[i],R)))
            self.b_u_hat_nat.append(np.zeros((ndims[i],R)))

            # self.rnd_m_u.append(rnd_m_u_mode)
            # self.m_u_hat_nat.append(m_u_hat_nat_mode)
            # self.v_u_hat_nat.append(v_u_hat_nat_mode)
            # self.a_u_hat_nat.append(a_u_hat_nat_mode)
            # self.b_u_hat_nat.append(b_u_hat_nat_mode)



    def get_initial_params(self):

        m_w = []
        v_w = []
        for i in range(len(self.rnd_m_w)):
            m_w.append(self.rnd_m_w[ i ])
            v_w.append(1.0 / self.v_w_hat_nat[ i ])
        
        m_u = []
        v_u = []
        for i in range(len(self.rnd_m_u)):
            m_u.append(self.rnd_m_u[ i ])
            v_u.append(1.0 /self.v_u_hat_nat[ i ])


        return { 'm_w': m_w, 'v_w': v_w ,'m_u':m_u,'v_u':v_u,'a': self.m_sigma,
            'b': self.v_sigma }

        # return { 'm_w': m_w, 'v_w': v_w , 'a': self.a_sigma_hat_nat + 1,
        #     'b': -self.b_sigma_hat_nat }


    def get_params(self):

        m_w = []
        v_w = []
        p_w = []
        for i in range(len(self.rnd_m_w)):
            m_w.append(self.m_w_hat_nat[ i ] / self.v_w_hat_nat[ i ])
            v_w.append(1.0 / self.v_w_hat_nat[ i ])
            p_w.append( self.rho_w_hat_nat[ i ])

        m_u = []
        v_u = []
        for i in range(len(self.rnd_m_u)):
            m_u.append(self.m_u_hat_nat[ i ] / self.v_u_hat_nat[ i ])
            v_u.append(1.0 /self.v_u_hat_nat[ i ])

        return { 'm_w': m_w, 'v_w': v_w ,'p_w': p_w,'m_u':m_u,'v_u':v_u,'a': self.m_sigma,
            'b': self.v_sigma }
 
    #going back to update the apprx. factors for weight and embedding  
    #params represent the current posterior
    #self.xxx represent the prior approx. factors

    def gauss_pdf(self,x,mean,var):
        pdf = 1.0 / np.sqrt(2 * math.pi*np.abs(var)) * np.exp((-0.5 * (x-mean)**2)/var)
        return pdf

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def inverse_sigmoid(self,x):
        return np.log(x/(1-x))


    '''
    def refine_prior(self, params):


        # factors for weight
        for i in range(len(params[ 'm_w' ])):
            # for j in range(params[ 'm_w' ][ i ].shape[ 0 ]):
            #     for k in range(params[ 'm_w' ][ i ].shape[ 1 ]):

                    # We obtain the parameters of the cavity distribution

            v_w_nat = 1.0 / params[ 'v_w' ][ i ]
            m_w_nat = params[ 'm_w' ][ i ] / \
                params[ 'v_w' ][ i ]

            # factors 
            v_w_cav_nat = v_w_nat - self.v_w_hat_nat[ i ]
            m_w_cav_nat = m_w_nat - self.m_w_hat_nat[ i ]

            # print('v_w_cav_nat',v_w_cav_nat.mean())
            # print('m_w_cav_nat',m_w_cav_nat.mean())

            v_w_cav = 1.0 / (v_w_cav_nat+1e-4)
            m_w_cav = m_w_cav_nat / (v_w_cav_nat + 1e-4)

            # v_w_cav = 1.0 / (v_w_cav_nat)
            # m_w_cav = m_w_cav_nat / (v_w_cav_nat)

            # print('v_w_cav',v_w_cav.mean())
            # print('m_w_cav',m_w_cav.mean())

            # update rho^star
            rho_star = np.log(self.gauss_pdf(m_w_cav,0.0,v_w_cav+self.tau_0)/self.gauss_pdf(m_w_cav,0.0,v_w_cav))

            v_w_til = 1/(v_w_cav_nat + 1/self.tau_0)
            m_w_til =  v_w_til*(m_w_cav/v_w_cav)
            # print('rho_star',rho_star.mean())

            # print('v_w',v_w_til.mean())
            # print('m_w',m_w_til.mean())

            rho_til = rho_star + self.inverse_sigmoid(self.rho_0)

            # print('rho_til',rho_til.mean())

            m_w_new = self.sigmoid(rho_til)*m_w_til
            v_w_new = self.sigmoid(rho_til)*(v_w_til + (1-self.sigmoid(rho_til))*m_w_til**2)

            # print('v_w_new',v_w_new.mean())
            # print('m_w_new',m_w_new.mean())
            # a_w_nat = self.a_w - 1
            # b_w_nat = -self.b_w
            # a_w_cav_nat = a_w_nat - self.a_w_hat_nat[ i ][ j, k ]
            # b_w_cav_nat = b_w_nat - self.b_w_hat_nat[ i ][ j, k ]
            # a_w_cav = a_w_cav_nat + 1
            # b_w_cav = -b_w_cav_nat

            v_w_new_nat = 1.0 / v_w_new
            m_w_new_nat = m_w_new / v_w_new

            # valid update check
            
            # self.m_w_hat_nat[ i ] = m_w_new_nat -  m_w_cav_nat
            # self.v_w_hat_nat[ i ] = v_w_new_nat - v_w_cav_nat 
            # self.rho_w_hat_nat[ i ] = rho_star
            
            # params[ 'm_w' ][ i ] = m_w_new
            # params[ 'v_w' ][ i ] = v_w_new   


            
            
            index1 = np.logical_or.reduce((v_w_cav > 0, v_w_cav < 1e6,v_w_new > 0, ~np.isnan(m_w_new), ~np.isnan(v_w_new)))
            index2 = np.logical_or.reduce((v_w_new < 1e5,~np.isinf(rho_star), ~np.isinf(m_w_new)))

            final_index = np.logical_or(index1,index2)

            # print('index_len', final_index)

            self.m_w_hat_nat[ i ] = np.where(~final_index,self.m_w_hat_nat[ i ],m_w_new_nat - \
                    m_w_cav_nat)
            self.v_w_hat_nat[ i ] = np.where(~final_index,self.v_w_hat_nat[ i ],v_w_new_nat - \
                    v_w_cav_nat)  
            self.rho_w_hat_nat[ i ] = np.where(~final_index,self.rho_w_hat_nat[ i ],rho_star)
            
            params[ 'm_w' ][ i ] = np.where(final_index,params[ 'm_w' ][ i ],m_w_new)
            params[ 'v_w' ][ i ] = np.where(final_index,params[ 'v_w' ][ i ],v_w_new)        
            
            # print(rho_star)

            # if v_w_cav > 0  and v_w_cav < 1e6 and v_w_new > 0 and v_w_new < 1e6 \
            #     and ~np.isnan(m_w_new) and ~np.isnan(v_w_new) and ~np.isinf(rho_star) and ~np.isinf(m_w_new):

            #     v_w_new_nat = 1.0 / v_w_new
            #     m_w_new_nat = m_w_new / v_w_new

            # if v_w_cav > 0 and b_w_cav > 0 and a_w_cav > 1 and \
            #     v_w_cav < 1e6:

                # We obtain the values of the new parameters of the
                # posterior approximation


                # print('m_w_new_nat',m_w_new_nat)
                # print('m_w_cav_nat',m_w_cav_nat)
                # self.m_w_hat_nat[ i ] = m_w_new_nat - \
                #     m_w_cav_nat
                # self.v_w_hat_nat[ i ] = v_w_new_nat - \
                #     v_w_cav_nat
                # self.rho_w_hat_nat[ i ] = rho_star


                # We update the posterior approximation

                # params[ 'm_w' ][ i ] = m_w_new
                # params[ 'v_w' ][ i ] = v_w_new

                        # self.a_w = a_w_new
                        # self.b_w = b_w_new

        # # factors for embedding
        # for i in range(len(params[ 'm_u' ])):
        #     for j in range(params[ 'm_u' ][ i ].shape[ 0 ]):
        #         for k in range(params[ 'm_u' ][ i ].shape[ 1 ]):

        #             # We obtain the parameters of the cavity distribution

        #             v_w_nat = 1.0 / params[ 'v_u' ][ i ][ j, k ]
        #             m_w_nat = params[ 'm_u' ][ i ][ j, k ] / \
        #                 params[ 'v_u' ][ i ][ j, k ]
        #             v_w_cav_nat = v_w_nat - self.v_u_hat_nat[ i ][ j, k ]
        #             m_w_cav_nat = m_w_nat - self.m_u_hat_nat[ i ][ j, k ]
        #             v_w_cav = 1.0 / v_w_cav_nat
        #             m_w_cav = m_w_cav_nat / v_w_cav_nat
        #             a_w_nat = self.a_u - 1
        #             b_w_nat = -self.b_u
        #             a_w_cav_nat = a_w_nat - self.a_u_hat_nat[ i ][ j, k ]
        #             b_w_cav_nat = b_w_nat - self.b_u_hat_nat[ i ][ j, k ]
        #             a_w_cav = a_w_cav_nat + 1
        #             b_w_cav = -b_w_cav_nat

        #             if v_w_cav > 0 and b_w_cav > 0 and a_w_cav > 1 and \
        #                 v_w_cav < 1e6:

        #                 # We obtain the values of the new parameters of the
        #                 # posterior approximation

        #                 v = v_w_cav + b_w_cav / (a_w_cav - 1)
        #                 v1  = v_w_cav + b_w_cav / a_w_cav
        #                 v2  = v_w_cav + b_w_cav / (a_w_cav + 1)
        #                 logZ = -0.5 * np.log(v) - 0.5 * m_w_cav**2 / v
        #                 logZ1 = -0.5 * np.log(v1) - 0.5 * m_w_cav**2 / v1
        #                 logZ2 = -0.5 * np.log(v2) - 0.5 * m_w_cav**2 / v2
        #                 d_logZ_d_m_w_cav = -m_w_cav / v
        #                 d_logZ_d_v_w_cav = -0.5 / v + 0.5 * m_w_cav**2 / v**2

        #                 m_w_new = m_w_cav + v_w_cav * d_logZ_d_m_w_cav
        #                 v_w_new = v_w_cav - v_w_cav**2 * \
        #                     (d_logZ_d_m_w_cav**2 - 2 * d_logZ_d_v_w_cav)

        #                 a_w_new = 1.0 / (np.exp(logZ2 - 2 * logZ1 + logZ) * \
        #                     (a_w_cav + 1) / a_w_cav - 1.0)
        #                 b_w_new = 1.0 / (np.exp(logZ2 - logZ1) * \
        #                     (a_w_cav + 1) / (b_w_cav) - np.exp(logZ1 - \
        #                     logZ) * a_w_cav / b_w_cav)
        #                 v_w_new_nat = 1.0 / v_w_new
        #                 m_w_new_nat = m_w_new / v_w_new
        #                 a_w_new_nat = a_w_new - 1
        #                 b_w_new_nat = -b_w_new

        #                 # We update the parameters of the approximate factor,
        #                 # whih is given by the ratio of the new posterior
        #                 # approximation and the cavity distribution

        #                 self.m_u_hat_nat[ i ][ j, k ] = m_w_new_nat - \
        #                     m_w_cav_nat
        #                 self.v_u_hat_nat[ i ][ j, k ] = v_w_new_nat - \
        #                     v_w_cav_nat
        #                 self.a_u_hat_nat[ i ][ j, k ] = a_w_new_nat - \
        #                     a_w_cav_nat
        #                 self.b_u_hat_nat[ i ][ j, k ] = b_w_new_nat - \
        #                     b_w_cav_nat

        #                 # We update the posterior approximation

        #                 params[ 'm_u' ][ i ][ j, k ] = m_w_new
        #                 params[ 'v_u' ][ i ][ j, k ] = v_w_new

        #                 self.a_u = a_w_new
        #                 self.b_u = b_w_new
            

        return params
    
    '''


    
    def refine_prior(self, params):


        # factors for weight
        for i in range(len(params[ 'm_w' ])):
            for j in range(params[ 'm_w' ][ i ].shape[ 0 ]):
                for k in range(params[ 'm_w' ][ i ].shape[ 1 ]):

                    # We obtain the parameters of the cavity distribution

                    v_w_nat = 1.0 / params[ 'v_w' ][ i ][ j, k ]
                    m_w_nat = params[ 'm_w' ][ i ][ j, k ] / \
                        params[ 'v_w' ][ i ][ j, k ]

                    # factors 
                    v_w_cav_nat = v_w_nat - self.v_w_hat_nat[ i ][ j, k ]
                    m_w_cav_nat = m_w_nat - self.m_w_hat_nat[ i ][ j, k ]

                    v_w_cav = 1.0 / v_w_cav_nat
                    m_w_cav = m_w_cav_nat / v_w_cav_nat

                    # update rho^star
                    rho_star = np.log(self.gauss_pdf(m_w_cav,0.0,v_w_cav+self.tau_0)/self.gauss_pdf(m_w_cav,0.0,v_w_cav))

                    v_w_til = 1/(v_w_cav_nat + 1/self.tau_0)
                    m_w_til =  v_w_til*(m_w_cav/v_w_cav)
                    rho_til = rho_star + self.inverse_sigmoid(self.rho_0)

                    m_w_new = self.sigmoid(rho_til)*m_w_til
                    v_w_new = self.sigmoid(rho_til)*(v_w_til + (1-self.sigmoid(rho_til))*m_w_til**2)

                    # a_w_nat = self.a_w - 1
                    # b_w_nat = -self.b_w
                    # a_w_cav_nat = a_w_nat - self.a_w_hat_nat[ i ][ j, k ]
                    # b_w_cav_nat = b_w_nat - self.b_w_hat_nat[ i ][ j, k ]
                    # a_w_cav = a_w_cav_nat + 1
                    # b_w_cav = -b_w_cav_nat

                    # valid update check
                    if v_w_cav > 0  and v_w_cav < 1e6 and v_w_new > 0 and v_w_new < 1e6 \
                        and ~np.isnan(m_w_new) and ~np.isnan(v_w_new) and ~np.isinf(rho_star) and ~np.isinf(m_w_new):
                        v_w_new_nat = 1.0 / v_w_new
                        m_w_new_nat = m_w_new / v_w_new

                    # if v_w_cav > 0 and b_w_cav > 0 and a_w_cav > 1 and \
                    #     v_w_cav < 1e6:

                        # We obtain the values of the new parameters of the
                        # posterior approximation


                        # print('m_w_new_nat',m_w_new_nat)
                        # print('m_w_cav_nat',m_w_cav_nat)
                        self.m_w_hat_nat[ i ][ j, k ] = m_w_new_nat - \
                            m_w_cav_nat
                        self.v_w_hat_nat[ i ][ j, k ] = v_w_new_nat - \
                            v_w_cav_nat
                        self.rho_w_hat_nat[ i ][ j, k ] = rho_star

                        # self.a_w_hat_nat[ i ][ j, k ] = a_w_new_nat - \
                        #     a_w_cav_nat
                        # self.b_w_hat_nat[ i ][ j, k ] = b_w_new_nat - \
                        #     b_w_cav_nat

                        # We update the posterior approximation

                        params[ 'm_w' ][ i ][ j, k ] = m_w_new
                        params[ 'v_w' ][ i ][ j, k ] = v_w_new

                        # self.a_w = a_w_new
                        # self.b_w = b_w_new

        # # factors for embedding
        # for i in range(len(params[ 'm_u' ])):
        #     for j in range(params[ 'm_u' ][ i ].shape[ 0 ]):
        #         for k in range(params[ 'm_u' ][ i ].shape[ 1 ]):

        #             # We obtain the parameters of the cavity distribution

        #             v_w_nat = 1.0 / params[ 'v_u' ][ i ][ j, k ]
        #             m_w_nat = params[ 'm_u' ][ i ][ j, k ] / \
        #                 params[ 'v_u' ][ i ][ j, k ]
        #             v_w_cav_nat = v_w_nat - self.v_u_hat_nat[ i ][ j, k ]
        #             m_w_cav_nat = m_w_nat - self.m_u_hat_nat[ i ][ j, k ]
        #             v_w_cav = 1.0 / v_w_cav_nat
        #             m_w_cav = m_w_cav_nat / v_w_cav_nat
        #             a_w_nat = self.a_u - 1
        #             b_w_nat = -self.b_u
        #             a_w_cav_nat = a_w_nat - self.a_u_hat_nat[ i ][ j, k ]
        #             b_w_cav_nat = b_w_nat - self.b_u_hat_nat[ i ][ j, k ]
        #             a_w_cav = a_w_cav_nat + 1
        #             b_w_cav = -b_w_cav_nat

        #             if v_w_cav > 0 and b_w_cav > 0 and a_w_cav > 1 and \
        #                 v_w_cav < 1e6:

        #                 # We obtain the values of the new parameters of the
        #                 # posterior approximation

        #                 v = v_w_cav + b_w_cav / (a_w_cav - 1)
        #                 v1  = v_w_cav + b_w_cav / a_w_cav
        #                 v2  = v_w_cav + b_w_cav / (a_w_cav + 1)
        #                 logZ = -0.5 * np.log(v) - 0.5 * m_w_cav**2 / v
        #                 logZ1 = -0.5 * np.log(v1) - 0.5 * m_w_cav**2 / v1
        #                 logZ2 = -0.5 * np.log(v2) - 0.5 * m_w_cav**2 / v2
        #                 d_logZ_d_m_w_cav = -m_w_cav / v
        #                 d_logZ_d_v_w_cav = -0.5 / v + 0.5 * m_w_cav**2 / v**2

        #                 m_w_new = m_w_cav + v_w_cav * d_logZ_d_m_w_cav
        #                 v_w_new = v_w_cav - v_w_cav**2 * \
        #                     (d_logZ_d_m_w_cav**2 - 2 * d_logZ_d_v_w_cav)

        #                 a_w_new = 1.0 / (np.exp(logZ2 - 2 * logZ1 + logZ) * \
        #                     (a_w_cav + 1) / a_w_cav - 1.0)
        #                 b_w_new = 1.0 / (np.exp(logZ2 - logZ1) * \
        #                     (a_w_cav + 1) / (b_w_cav) - np.exp(logZ1 - \
        #                     logZ) * a_w_cav / b_w_cav)
        #                 v_w_new_nat = 1.0 / v_w_new
        #                 m_w_new_nat = m_w_new / v_w_new
        #                 a_w_new_nat = a_w_new - 1
        #                 b_w_new_nat = -b_w_new

        #                 # We update the parameters of the approximate factor,
        #                 # whih is given by the ratio of the new posterior
        #                 # approximation and the cavity distribution

        #                 self.m_u_hat_nat[ i ][ j, k ] = m_w_new_nat - \
        #                     m_w_cav_nat
        #                 self.v_u_hat_nat[ i ][ j, k ] = v_w_new_nat - \
        #                     v_w_cav_nat
        #                 self.a_u_hat_nat[ i ][ j, k ] = a_w_new_nat - \
        #                     a_w_cav_nat
        #                 self.b_u_hat_nat[ i ][ j, k ] = b_w_new_nat - \
        #                     b_w_cav_nat

        #                 # We update the posterior approximation

        #                 params[ 'm_u' ][ i ][ j, k ] = m_w_new
        #                 params[ 'v_u' ][ i ][ j, k ] = v_w_new

        #                 self.a_u = a_w_new
        #                 self.b_u = b_w_new
            

        return params
        
        