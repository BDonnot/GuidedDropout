"""
This file implements an interface more user friendly to use with the TensorflowHelpers package
"""

import copy

import pdb

import tensorflow as tf
from .GuidedDropout import SpecificGDOEncoding, SpecificGDCEncoding

from TensorflowHelpers import ComplexGraph, NNFully

#TODO refactor it with DenseLayer directly
class DenseLayerwithGD:
    def __init__(self, input, size,
                 relu=False, bias=True,
                 weight_normalization=False,
                 keep_prob=None, layernum=0,
                 guided_dropout_mask=None,
                 guided_dropconnect_mask=None):
        """
        Implement the guided dropout and guided dropconnect case.
        for weight normalization see https://arxiv.org/abs/1602.07868
        for counting the flops of operations see https://mediatum.ub.tum.de/doc/625604/625604
        :param input: input of the layer 
        :param size: layer size (number of outputs units)
        :param relu: do you use relu ?
        :param bias: do you add bias ?
        :param guided_dropconnect_mask: tensor of the mask matrix  #TODO 
        :param weight_normalization: do you use weight normalization (see https://arxiv.org/abs/1602.07868)
        :param keep_prob: a scalar tensor for dropout layer (None if you don't want to use it)
        :param layernum: number of layer (this layer in the graph)
        :param guided_dropout_mask: the mask use for guided dropout (None if you don't want it) [tensorflow tensor]
        :param guided_dropconnect_mask: the mask use for guided dropconnect ('None' if you don't want it) [tensorflow tensor]
        """
        nin_ = int(input.get_shape()[1])
        self.nbparams = 0  # number of trainable parameters
        self.flops = 0  # flops for a batch on 1 data
        self.input = input
        self.weightnormed = False
        self.bias = False
        self.res = None
        with tf.variable_scope("dense_layer_{}".format(layernum)):
            self.w_ = tf.get_variable(name="weights_matrix",
                                      shape=[nin_, size],
                                      initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
                                      # initializer=tf.get_default_graph().get_tensor_by_name(tf.get_variable_scope().name+"/weights_matrix:0"),
                                      trainable=True)  # weight matrix
            self.nbparams += int(nin_ * size)

            if weight_normalization:
                self.weightnormed = True
                self.g = tf.get_variable(shape=[size],
                                         name="weight_normalization_g",
                                         initializer=tf.constant_initializer(value=1.0, dtype="float32"),
                                         # initializer=tf.get_default_graph().get_tensor_by_name(tf.get_variable_scope().name+"/weight_normalization_g:0"),
                                         trainable=True)
                self.nbparams += int(size)
                self.scaled_matrix = tf.nn.l2_normalize(self.w_, dim=0, name="weight_normalization_scaled_matrix")
                self.flops += size * (
                2 * nin_ - 1)  # clomputation of ||v|| (size comptuation of inner product of vector of size nin_)
                self.flops += 2 * nin_ - 1  # division by ||v|| (matrix vector product)
                self.w = tf.multiply(self.scaled_matrix, self.g, name="weight_normalization_weights")
                self.flops += 2 * nin_ - 1  # multiplication by g (matrix vector product)
            else:
                self.w = self.w_

            if guided_dropconnect_mask is not None:
                self.w = tf.multiply(self.w, guided_dropconnect_mask, name="applying_guided_dropconnect")
                self.flops += nin_*size

            res = tf.matmul(self.input, self.w, name="multiplying_weight_matrix")
            self.flops += 2 * nin_ * size - size

            if bias:
                self.bias = True
                self.b = tf.get_variable(shape=[size],
                                         initializer=tf.constant_initializer(value=0.0, dtype="float32"),
                                         # initializer=tf.get_default_graph().get_tensor_by_name(tf.get_variable_scope().name+"/bias:0"),
                                         name="bias",
                                         trainable=True)
                self.nbparams += int(size)
                res = tf.add(res, self.b, name="adding_bias")
                self.flops += size  # vectors addition of size "size"

            if relu:
                res = tf.nn.relu(res, name="applying_relu")
                self.flops += size  # we consider relu of requiring 1 computation per number (one max)

            if guided_dropout_mask is not None:
                res = tf.multiply(res, guided_dropout_mask, name="applying_guided_dropout")
                self.flops += size

            if keep_prob is not None:
                res = tf.nn.dropout(res, keep_prob=keep_prob, name="applying_dropout")
                # we consider that generating random number count for 1 operation
                self.flops += size  # generate the "size" real random numbers
                self.flops += size  # building the 0-1 vector of size "size" (thresholding "size" random values)
                self.flops += size  # element wise multiplication with res
            self.res = res

    def initwn(self, sess, scale_init=1.0):
        """
        initialize the weight normalization as describe in https://arxiv.org/abs/1602.07868
        don't do anything if the weigth normalization have not been "activated"
        :param sess: the tensorflow session
        :param scale_init: the initial scale
        :return: 
        """
        if not self.weightnormed:
            return
        # input = sess.run(input)
        with tf.variable_scope("init_wn_layer"):
            m_init, v_init = sess.run(tf.nn.moments(tf.matmul(self.input, self.w), [0]))
            # pdb.set_trace()
            sess.run(tf.assign(self.g, scale_init/tf.sqrt(v_init + 1e-10), name="weigth_normalization_init_g"))
            if self.bias:
                sess.run(tf.assign(self.b, -m_init*scale_init, name="weigth_normalization_init_b"))


# TODO merge that with ComplexGraph
class ComplexGraphWithGD(ComplexGraph):
    def __init__(self, data,
                 outputsize,
                 sizes,
                 var_x_name={"input"}, var_y_name={"output"},
                 nnType=NNFully, argsNN=(), kwargsNN={},
                 encDecNN=NNFully, args_enc=(), kwargs_enc={},
                 args_dec=(), kwargs_dec={},
                 kwargs_enc_dec=None,
                 spec_encoding={},
                 dropout_spec={}, dropconnect_spec={}):
        """
        This class can deal with multiple input/output.
        It will first "encode" with a neural network of type "encDecNN" for each input.
        Then concatenate all the outputs to feed the "main" neural network of type "nnType"
        Afterwards, information goes through a decoding process, with neural network of type "encDecNN"

        The value for each can be retrieve with standard methods:
        - self.get_true_output_dict()
        - self.vars_out
        - self.vars_in

        Basically, this should represent the neural network.
        :param data: the dictionnary of input tensor data (key=name, value=tensorflow tensor)
        :param var_x_name: iterable: the names of all the input variables
        :param var_y_name: iterable: the name of  all the output variables
        :param nnType: the type of neural network to use
        :param args forwarded to the initializer of neural network
        :param kwargsNN: key word arguments forwarded to the initializer of neural network
        :param encDecNN: class to use to build the neural networks for encoding / decoding
        :param args_enc:
        :param kwargs_enc: 
        :param args_dec:
        :param kwargs_dec: 
        :param kwargs_enc_dec: 
        :param sizes: the size output by the encoder for each input variable. Dictionnary with key: variable names, value: size
        :param outputsize: the output size for the intermediate / main neural network
        :param dropout_spec
        :param dropconnect_spec
        """
        modifkwargsNN = copy.deepcopy(kwargsNN)
        if "layerClass" in modifkwargsNN:
            msg = "I: ComplexGraphWithGD: you pass a specific class \"{}\" to build the layers."
            msg += " This class should accept the key-word arguments \"guided_dropout_mask\" and \"guided_droconnect_mask\" "
            print(msg.format(modifkwargsNN["layerClass"]))
        else:
            modifkwargsNN["layerClass"] = DenseLayerwithGD

        dict_kargs = {}
        if "kwardslayer" in modifkwargsNN:
            dict_kargs = copy.deepcopy(modifkwargsNN["kwardslayer"])

        if "guided_dropout_mask" in dict_kargs:
            msg = "W: \"guided_dropout_mask\" in \"kwargsNN\" of object ComplexGraphWithGD will be erased"
            msg += " and replace with the data in \"var_dropout_name\""
            print(msg)
        if "guided_dropconnect_mask" in dict_kargs:
            msg = "W: \"guided_droconnect_mask\" in \"kwargsNN\" of object ComplexGraphWithGD will be erased"
            msg += " and replace with the data in \"var_dropconnect_name\""
            print(msg)

        if len(dropout_spec):
            # for var in var_dropout_name:
                #TODO HEre for multiple variable in guided dropout (logical or and better choice of mask!)
            vars = sorted(list(dropout_spec["vars"]))
            var = vars[0]
            if len(dropout_spec) > 1:
                msg = "W guided dropout with multiple masks is for now not coded."
                msg += " Only mask corresponding to variable {}"
                msg += " will be used."
                print(msg.format(var))
            if not "proba_select" in dropout_spec:
                msg = "Error: ComplexGraphWithGD: the fields \"proba_select\" in \"dropout_spec\" is not specified"
                raise RuntimeError(msg)
            proba_select = dropout_spec["proba_select"]
            self.encgd = SpecificGDOEncoding(sizeinputonehot=int(data[var].get_shape()[1]),
                                             sizeout=outputsize,
                                             proba_select=proba_select)
            dict_kargs['guided_dropout_mask'] = self.encgd(data[var])

        if len(dropconnect_spec):
            # for var in var_dropout_name:
                #TODO HEre for multiple variable in guided dropout (logical or and better choice of mask!)
            vars = sorted(list(dropconnect_spec["vars"]))
            var = vars[0]
            if len(dropconnect_spec) > 1:
                msg = "W guided dropconnect with multiple masks is for now not coded."
                msg += " Only mask corresponding to variable {}"
                msg += " will be used."
                print(msg.format(var))
            if not "proba_select" in dropconnect_spec:
                msg = "Error: ComplexGraphWithGD: the fields \"proba_select\" in \"dropconnect_spec\" is not specified"
                raise RuntimeError(msg)
            proba_select = dropconnect_spec["proba_select"]
            if not "nrow" in dropconnect_spec:
                msg = "Error: ComplexGraphWithGD: the fields \"nrow\" in \"dropconnect_spec\" is not specified"
                raise RuntimeError(msg)
            nrow = dropconnect_spec["nrow"]
            self.encgd = SpecificGDCEncoding(sizeinputonehot=int(data[var].get_shape()[1]),
                                             ncol=outputsize,
                                             nrow=nrow,
                                             proba_select=proba_select)
            dict_kargs['guided_dropconnect_mask'] = self.encgd(data[var])

        modifkwargsNN["kwardslayer"] = dict_kargs
        ComplexGraph.__init__(self,
                              data=data,
                              outputsize=outputsize, sizes=sizes,
                              var_x_name=var_x_name, var_y_name=var_y_name,
                              nnType=nnType, argsNN=argsNN, kwargsNN=modifkwargsNN,
                              encDecNN=encDecNN, args_enc=args_enc, kwargs_enc=kwargs_enc,
                              args_dec=args_dec, kwargs_dec=kwargs_dec, kwargs_enc_dec=kwargs_enc_dec,
                              spec_encoding=spec_encoding)