'''
Build a attention-based neural machine translation model
'''
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

#from ipdb import set_trace as dbg

import cPickle as pkl
import numpy
import numpy as np
import copy

import os
import warnings
import sys
import time

from collections import OrderedDict

from dataIter import Batch_data_from_file_iter, prepare_data

import theano.printing as printing

profile = False


# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]

    
# p is the probability of keeping a unit
def dropout_layer(state_before, use_noise, trng, p=0.5):
    proj = tensor.switch(
        use_noise,
        state_before * trng.binomial(state_before.shape, p=p, n=1,
                                     dtype=state_before.dtype),
        state_before * p)
    return proj


# make prefix-appended name
def _p(pp, name):
    return '%s_%s' % (pp, name)


# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            warnings.warn('%s is not in the archive' % kk)
            continue
        params[kk] = pp[kk]

    return params


# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'ff_nb': ('param_init_fflayer_nb', 'fflayer_nb'),
          'lstm': ('param_init_lstm', 'lstm_layer'),
          'gru': ('param_init_gru', 'gru_layer'),
          'gru_rec': ('param_init_gru', 'gru_layer_rec'),
          'rnn': ('param_init_rnn', 'rnn_layer'),
          }


def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


# some utilities
def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')


def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')


def tanh(x):
    return tensor.tanh(x)


def linear(x):
    return x


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out


# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None, ortho=True):
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')

    return params


def fflayer(tparams, state_below, options, prefix='rconv', activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')])


# feedforward layer with no bias: affine transformation + point-wise nonlinearity
def param_init_fflayer_nb(options, params, prefix='ff_nb', nin=None, nout=None, ortho=True):
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)

    return params


def fflayer_nb(tparams, state_below, options, prefix='ff_nb', activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(tensor.dot(state_below, tparams[_p(prefix, 'W')]))


# RNN layer
def param_init_rnn(options, params, prefix='rnn', nin=None, dim=None):
    Wx = norm_weight(nin, dim)
    params[_p(prefix, 'Wx')] = Wx
    Ux = ortho_weight(dim)
    params[_p(prefix, 'Ux')] = Ux
    params[_p(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')

    return params


def rnn_layer(tparams, state_below, options, prefix='rnn', mask=None, **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[_p(prefix, 'Ux')].shape[0]

    if mask == None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]

    def _step(m_, xx_, h_, Ux):
        preactx = tensor.dot(h_, Ux)
        preactx = preactx + xx_

        h = tensor.tanh(preactx)

        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h  # , r, u, preact, preactx

    rval, updates = theano.scan(_step,
                                sequences=[mask, state_belowx],
                                outputs_info=[tensor.alloc(0., n_samples, dim)],
                                non_sequences=[tparams[_p(prefix, 'Ux')]],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile)
    rval = [rval]
    return rval


# GRU layer
def param_init_gru(options, params, prefix='gru', nin=None, dim=None, hiero=False):
    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix, 'U')] = U

    Wx = norm_weight(nin, dim)
    params[_p(prefix, 'Wx')] = Wx
    Ux = ortho_weight(dim)
    params[_p(prefix, 'Ux')] = Ux
    params[_p(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')

    if options['model_version'] == 'gru_rec':
        params[_p(prefix, 'b_rec')] = numpy.zeros((dim,)).astype('float32')

    return params


def gru_layer(tparams, state_below, options, prefix='gru', mask=None, one_step=False, init_state=None, **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    if one_step:
        assert init_state, 'previous state must be provided'

    dim = tparams[_p(prefix, 'Ux')].shape[1]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]

    def _step(m_, x_, xx_, h_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_

        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        h = tensor.tanh(preactx)

        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h  # , r, u, preact, preactx

    seqs = [mask, state_below_, state_belowx]

    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Ux')]]

    if one_step:
        rval = _step(*(seqs + [init_state] + shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state],
                                    non_sequences=shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    rval = [rval]
    return rval


def gru_layer_rec(tparams, state_below, options, prefix='gru', mask=None, one_step=False, init_state=None, **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    if one_step:
        assert init_state, 'previous state must be provided'

    dim = tparams[_p(prefix, 'Ux')].shape[1]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]

    def _step(m_, x_, xx_, h_, h_rec, U, Ux, b_rec):
        preact = tensor.dot(h_, U)
        preact += x_

        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        preactx_rec = tensor.dot(h_, Ux)
        preactx = preactx_rec * r
        preactx = preactx + xx_

        h_rec = tensor.tanh(preactx_rec + b_rec)
        h = tensor.tanh(preactx)

        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, h_rec  # , r, u, preact, preactx

    seqs = [mask, state_below_, state_belowx]

    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Ux')],
                   tparams[_p(prefix, 'b_rec')]]

    if one_step:
        rval = _step(*(seqs + [init_state, tensor.alloc(0., dim, dim)] + shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state, tensor.alloc(0., n_samples, dim)],
                                    non_sequences=shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    return rval


# LSTM layer
def param_init_lstm(options, params, prefix='lstm', nin=None, dim=None, hiero=False):
    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim),
                           norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix, 'U')] = U
    bias_b = numpy.zeros((4 * dim,)).astype('float32')
    bias_b[dim * 3:dim * 4] = 1.0
    params[_p(prefix, 'b')] = bias_b

    return params


def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None, **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[_p(prefix, 'U')].shape[0]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_
        preact += tparams[_p(prefix, 'b')]

        i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
        o = tensor.nnet.sigmoid(_slice(preact, 2, dim))
        c = tensor.tanh(_slice(preact, 3, dim))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c, i, f, o, preact

    state_below = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]

    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(0., n_samples, dim),
                                              tensor.alloc(0., n_samples, dim),
                                              None, None, None, None],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile)

    return rval


# initialize all parameters
def init_params(options):
    numpy.random.seed(1234)
    params = OrderedDict()
    # embedding
    params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])

    # encoder: LSTM
    params = get_layer(options['model_version'])[0](options, params, prefix=options['model_version'],
                                                    nin=options['dim_word'], dim=options['dim'])

    # readout
    params = get_layer('ff')[0](options, params, prefix='ff_logit', nin=options['dim'], nout=options['n_words'], ortho=False)

    if options['model_version'] == 'gru_rec':
        params = get_layer('ff')[0](options, params, prefix='rec_', nin=options['dim'], nout=options['dim'], ortho=False)

    return params


# build a training model
def build_model(tparams, options):
    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    x_mask = tensor.matrix('x_mask', dtype='float32')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]


    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])

    #emb = printing.Print('text')(emb)
    #theano.printing.debugprint(emb, file=open('emb.txt', 'w'))

    emb_shifted = tensor.zeros_like(emb)
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    if options['use_word_dropout']:
        emb_shifted = dropout_layer(emb_shifted, use_noise, trng, p=1.0-options['use_word_dropout_p'])

    #emb_shifted = printing.Print('text_emb_shifted')(emb_shifted)
    #x_mask  = printing.Print('text_x_mask')(x_mask)

    proj = get_layer(options['model_version'])[1](tparams, emb_shifted, options,
                                                  prefix=options['model_version'],
                                                  mask=x_mask,
                                                  one_step=False)

    proj_h = proj[0]
    if options['use_word_dropout']:
        proj_h = dropout_layer(proj_h, use_noise, trng, p=1.0-options['use_word_dropout_p'])
    if options['model_version'] == 'gru_rec':
        target_rec = proj[0]  # which is the same as proj_h but no dropout
        proj_h_rec = proj[1]
        logit_rec = get_layer('ff')[1](tparams, proj_h_rec, options, prefix='rec_', activ='linear')

    # compute word probabilities
    logit = get_layer('ff')[1](tparams, proj_h, options, prefix='ff_logit', activ='linear')
    logit_shp = logit.shape
    probs = tensor.nnet.softmax(logit.reshape([logit_shp[0] * logit_shp[1], logit_shp[2]]))

    # reconstruction cost
    if options['model_version'] == 'gru_rec':
        proj_h_shifted = tensor.zeros_like(target_rec)
        proj_h_shifted = tensor.set_subtensor(proj_h_shifted[1:], target_rec[:-1])
        mse_mat = tensor.sqr(logit_rec - proj_h_shifted).sum(2) * x_mask
        temp_mse = mse_mat.sum(0)
        nb_word_vec = x_mask.sum(0)
        cost_mse_rec = (temp_mse / nb_word_vec).sum()

    # cost
    x_flat = x.flatten()
    x_flat_idx = tensor.arange(x_flat.shape[0]) * options['n_words'] + x_flat
    cost = -tensor.log(probs.flatten()[x_flat_idx])
    cost = cost.reshape([x.shape[0], x.shape[1]])
    cost = (cost * x_mask).sum(0)
    cost_test = cost
    if options['model_version'] == 'gru_rec':
        cost += options['rec_coeff'] * cost_mse_rec

    return trng, use_noise, x, x_mask, cost, cost_test


def pred_probs(f_log_probs, options, iterator, verbose=False):
    probs = []

    n_done = 0
    total_count = 0
    for x_raw in iterator:
        n_done += len(x_raw)

        x, x_mask = prepare_data(x_raw)

        total_count += x_mask.sum()

        if x is None:
            continue

        pprobs = f_log_probs(x, x_mask)
        for pp in pprobs:
            probs.append(pp)

        if verbose:
            print >> sys.stderr, '%d samples computed' % (n_done)

    return numpy.array(probs).sum()/total_count


# optimizers
# name(hyperp, tparams, grads, inputs (list), cost) = f_grad_shared, f_update
def adam(lr, tparams, grads, inp, cost):
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k) for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup, profile=profile)

    lr0 = 0.0002
    b1 = 0.1
    b2 = 0.001
    e = 1e-8

    updates = []

    i = theano.shared(numpy.float32(0.))
    i_t = i + 1.
    fix1 = 1. - b1**(i_t)
    fix2 = 1. - b2**(i_t)
    lr_t = lr0 * (tensor.sqrt(fix2) / fix1)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * tensor.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (tensor.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))

    f_update = theano.function([lr], [], updates=updates, on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, inp, cost):
    running_up2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rup2' % k) for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad2' % k) for k, p in tparams.iteritems()]

    rg2_new = [0.95 * rg2 + 0.05 * (g ** 2) for rg2, g in zip(running_grads2, grads)]
    rg2up = [(rg2, r_n) for rg2, r_n in zip(running_grads2, rg2_new)]

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg for zg, ru2, rg2 in zip(grads, running_up2, rg2_new)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(itemlist(tparams), updir)]

    inp += [lr]
    f_update = theano.function(inp, cost, updates=rg2up+ru2up+param_up, on_unused_input='ignore', profile=profile)

    return f_update


def debugging_adadelta(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_grad' % k) for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rup2' % k) for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad2' % k) for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rg2up, profile=profile)

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(itemlist(tparams), updir)]

    f_update = theano.function([lr], [], updates=ru2up+param_up, on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_grad' % k) for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad' % k) for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad2' % k) for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rgup+rg2up, profile=profile)

    updir = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_updir' % k) for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4)) for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads, running_grads2)]
    param_up = [(p, p + udn[1]) for p, udn in zip(itemlist(tparams), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new+param_up, on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def sgd(lr, tparams, grads, x, mask, y, cost):
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k) for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup, profile=profile)

    pup = [(p, p - lr * g) for p, g in zip(itemlist(tparams), gshared)]
    f_update = theano.function([lr], [], updates=pup, profile=profile)

    return f_grad_shared, f_update


def train(dim_word=100,  # word vector dimensionality
          dim=1000,  # the number of LSTM units
          model_version='gru',
          patience=10,
          max_epochs=5000,
          decay_c=0.,
          diag_c=0.,
          lrate=0.01,
          n_words=100000,
          optimizer='rmsprop',
          batch_size=16,
          valid_batch_size=16,
          saveto='model.npz',
          trainsetPath='test.txt',
          validsetPath='test.txt',
          testsetPath='test.txt',
          reload_=False,
          clip_c=1.,
          rec_coeff=0.1,
          use_word_dropout=True,
          use_word_dropout_p=0.5):

        # removed
        # encoder='gru',  ---> replaced by model_version
        # decoder='gru_cond',
        # n_words_src=1000000
        # dictionary_src=None,  # word dictionary
        # dictionary=None,  # word dictionary
        # hiero=None,  # 'gru_hiero', # or None
        # use_dropout=False,
        # euclidean_coeff=0.1,
        # covVec_in_attention=False,
        # covVec_in_decoder=False,
        # covVec_in_pred=False,
        # alpha_c=0.,
        # maxlen=100,  # maximum length of the description
        # validFreq=1000,
        # dispFreq=100
        # saveFreq=1000,  # save the parameters after every saveFreq updates
        # sampleFreq=100,  # generate some samples after every sampleFreq updates

    # Model options
    model_options = locals().copy()

    # reload options
    if reload_ and os.path.exists(saveto):
        with open('%s.pkl' % saveto, 'rb') as f:
            models_options = pkl.load(f)

    # import ipdb; ipdb.set_trace()
    print 'Loading data'
    train = Batch_data_from_file_iter(trainsetPath, batch_size)
    valid = Batch_data_from_file_iter(validsetPath, batch_size)
    test = Batch_data_from_file_iter(testsetPath, batch_size)

    print 'Building model'
    params = init_params(model_options)
    # reload parameters
    if reload_ and os.path.exists(saveto):
        params = load_params(saveto, params)

    tparams = init_tparams(params)

    trng, use_noise, x, x_mask, cost, cost_test = build_model(tparams, model_options)
    inps = [x, x_mask]

    # theano.printing.debugprint(cost.mean(), file=open('cost.txt', 'w'))

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost_test, profile=profile)
    print 'Done'

    cost = cost.mean()

    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # after any regularizer
    #print 'Building f_cost...',
    #f_cost = theano.function(inps, cost, profile=profile)
    #print 'Done'

    print 'Computing gradient...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    print 'Done'
    print 'Building f_grad...',
    f_grad = theano.function(inps, grads, profile=profile)
    print 'Done'

    # Cliping gradients
    if clip_c > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c**2),
                                           g / tensor.sqrt(g2) * clip_c,
                                           g))
        grads = new_grads

    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    # f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost)
    f_update = eval(optimizer)(lr, tparams, grads, inps, cost)
    print 'Done'

    print 'Optimization'

    history_errs = []
    # reload history
    if reload_ and os.path.exists(saveto):
        history_errs = list(numpy.load(saveto)['history_errs'])
    best_p = None
    bad_count = 0

    uidx = 0
    estop = False
    for eidx in xrange(max_epochs):
        # import ipdb; ipdb.set_trace()
        use_noise.set_value(1.)
        n_samples = 0
        for x_raw in train:
            n_samples += len(x_raw)
            uidx += 1

            x, x_mask = prepare_data(x_raw)

            ud_start = time.time()
            # cost = f_grad_shared(x, x_mask, y, y_mask)
            # f_update(lrate)
            cost = f_update(x, x_mask, lrate)
            ud = time.time() - ud_start

            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.

        print 'Epoch ', eidx + 1, 'UD ', ud

        # if numpy.mod(uidx, saveFreq) == 0:
        print 'Saving...',
        params = unzip(tparams)

        saveto_list = saveto.split('/')
        saveto_list[-1] = 'epoch' + str(eidx+1) + '_' + 'nbUpd' + str(uidx) + '_' + saveto_list[-1]
        saveName = '/'.join(saveto_list)
        numpy.savez(saveName, history_errs=history_errs, **params)
        pkl.dump(model_options, open('%s.pkl' % saveName, 'wb'))
        print 'Done'

        # if numpy.mod(uidx, validFreq) == 0:
        train_err = 0
        valid_err = 0
        test_err = 0
        use_noise.set_value(0.)

        if valid is not None:
            valid_err = pred_probs(f_log_probs, model_options, valid)  # .mean()
        if test is not None:
            test_err = pred_probs(f_log_probs, model_options, test)  # .mean()

        history_errs.append([valid_err, test_err])
        print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err
        print 'Seen %d samples' % n_samples

        if uidx == 0 or valid_err <= numpy.array(history_errs)[:, 0].min():
            best_p = unzip(tparams)
            bad_counter = 0
        if len(history_errs) > patience and valid_err >= numpy.array(history_errs)[:, 0].min():
            bad_counter += 1
            if bad_counter > patience:
                #import ipdb; ipdb.set_trace()
                print 'Early Stop!'
                estop = True
                break

        if estop:
            break

    if best_p is not None:
        zipp(best_p, tparams)

    train_err = 0
    valid_err = 0
    test_err = 0
    use_noise.set_value(0.)
    # train_err = pred_error(f_pred, prepare_data, train, kf)
    train_err = pred_probs(f_log_probs, model_options, train) 
    if valid is not None:
        valid_err = pred_probs(f_log_probs, model_options, valid)  # .mean()
    if test is not None:
        test_err = pred_probs(f_log_probs, model_options, test)  # .mean()

    print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err

    if best_p is not None:
        params = copy.copy(best_p)
    else:
        params = unzip(tparams)
    numpy.savez(saveto, zipped_params=best_p, train_err=train_err,
                valid_err=valid_err, test_err=test_err, history_errs=history_errs,
                **params)

    return train_err, valid_err, test_err


if __name__ == '__main__':
    pass
