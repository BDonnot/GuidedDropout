"""
This file implements an interface more user friendly to use with the TensorflowHelpers package
"""

import copy

import pdb

import tensorflow as tf
from .GuidedDropout import SpecificGDOEncoding, SpecificGDCEncoding, DTYPE_USED

from TensorflowHelpers import ExpGraph, ComplexGraph, NNFully



#TODO refactor it with DenseLayer directly
class DenseLayerwithGD:
    def __init__(self, input, size,
                 relu=False, bias=True,
                 weight_normalization=False,
                 keep_prob=None, layernum=0,
                 guided_dropout_mask=None,
                 guided_dropconnect_mask=None
    ):
        """
        Implement the guided dropout and guided dropconnect case.
        for weight normalization see https://arxiv.org/abs/1602.07868
        for counting the flops of operations see https://mediatum.ub.tum.de/doc/625604/625604
        :param input: input of the layer 
        :param size: size of the latent space (which is also the size of the mask)
        :param relu: do you use relu ?
        :param bias: do you add bias ?
        :param guided_dropconnect_mask: tensor of the mask matrix  #TODO 
        :param weight_normalization: do you use weight normalization (see https://arxiv.org/abs/1602.07868)
        :param keep_prob: a scalar tensor for dropout layer (None if you don't want to use it)
        :param layernum: number of layer (this layer in the graph)
        :param guided_dropout_mask: the mask use for guided dropout (None if you don't want it) [tensorflow tensor]
        :param guided_dropconnect_mask: the mask use for guided dropconnect ('None' if you don't want it) [tensorflow tensor]
        # :param kwardslayer: TODO
        """
        nin_ = int(input.get_shape()[1])
        self.nbparams = 0  # number of trainable parameters
        self.flops = 0  # flops for a batch on 1 data
        self.input = input
        self.weightnormed = False
        self.bias = bias
        self.res = None
        # guided_dropconnect_mask = kwardslayer["guided_dropconnect_mask"]
        # guided_dropout_mask = kwardslayer["guided_dropout_mask"]
        if guided_dropout_mask is not None or guided_dropconnect_mask is not None:
            name_layer = "LSI_block_{}".format(layernum)
        else:
            name_layer = "residual_block_{}".format(layernum)

        with tf.variable_scope(name_layer):
            self.w_e_fp32 = tf.get_variable(name="weights_matrix_enc",
                                      shape=[nin_, size],
                                      initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
                                      # initializer=tf.get_default_graph().get_tensor_by_name(tf.get_variable_scope().name+"/weights_matrix:0"),
                                      trainable=True,
                                      dtype=tf.float32)
            if DTYPE_USED != tf.float32:
                self.w_e = tf.cast(self.w_e_fp32, DTYPE_USED)
            else:
                self.w_e = self.w_e_fp32
            self.nbparams += int(nin_ * size)
            self.w_d_fp32 = tf.get_variable(name="weights_matrix_dec",
                                      shape=[size, nin_],
                                      initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
                                      # initializer=tf.get_default_graph().get_tensor_by_name(tf.get_variable_scope().name+"/weights_matrix:0"),
                                      trainable=True,
                                      dtype=tf.float32)
            if DTYPE_USED != tf.float32:
                self.w_d = tf.cast(self.w_d_fp32, DTYPE_USED)
            else:
                self.w_d = self.w_d_fp32

            self.nbparams += int(nin_ * size)

            if weight_normalization:
                raise RuntimeError("weight normalization not implemented yet")
            #     self.weightnormed = True
            #     self.g = tf.get_variable(shape=[size],
            #                              name="weight_normalization_g",
            #                              initializer=tf.constant_initializer(value=0.0, dtype=tf.float32),
            #                              # initializer=tf.get_default_graph().get_tensor_by_name(tf.get_variable_scope().name+"/weight_normalization_g:0"),
            #                              trainable=True,
            #                              dtype=tf.float32)
            #     if DTYPE_USED != tf.float32:
            #         self.g = tf.cast(self.g, DTYPE_USED)
            #
            #     self.nbparams += int(size)
            #     self.scaled_matrix = tf.nn.l2_normalize(self.w_, dim=0, name="weight_normalization_scaled_matrix")
            #     self.flops += size * (
            #     2 * nin_ - 1)  # clomputation of ||v|| (size comptuation of inner product of vector of size nin_)
            #     self.flops += 2 * nin_ - 1  # division by ||v|| (matrix vector product)
            #     self.w = tf.multiply(self.scaled_matrix, self.g, name="weight_normalization_weights")
            #     self.flops += 2 * nin_ - 1  # multiplication by g (matrix vector product)
            # else:
            #     self.w = self.w_e

            if guided_dropconnect_mask is not None:
                raise RuntimeError("guided dropconnect not implemented yet")
                # self.w = tf.multiply(self.w, guided_dropconnect_mask, name="applying_guided_dropconnect")
                # self.flops += nin_*size

            res = tf.matmul(self.input, self.w_e, name="multiplying_weight_matrix")
            self.flops += 2 * nin_ * size - size

            if bias:
                self.bias = True
                self.b = tf.get_variable(shape=[size],
                                         initializer=tf.constant_initializer(value=1.0, dtype=tf.float32),
                                         # initializer=tf.get_default_graph().get_tensor_by_name(tf.get_variable_scope().name+"/bias:0"),
                                         name="bias",
                                         trainable=True,
                                      dtype=tf.float32)
                if DTYPE_USED != tf.float32:
                    self.b = tf.cast(self.b, DTYPE_USED)

                self.nbparams += int(size)
                res = tf.add(res, self.b, name="adding_bias_e")
                self.flops += size  # vectors addition of size "size"

            self.after_layer = res  # the "encoded latent space" (after applying guided dropout, and before any non linearity)
            self.after_gd = res  # the "encoded latent space" (after applying guided dropout, and before any non linearity)

            if guided_dropout_mask is not None:
                res = tf.multiply(res, guided_dropout_mask, name="applying_guided_dropout")
                self.mask = guided_dropout_mask
                self.after_gd = res
                self.flops += size

            res = tf.matmul(res, self.w_d, name="out_latent_space")
            self.flops += 2 * nin_ * size - nin_

            self.res = tf.add(res, input, name="integrate_gd_modifications")
            self.flops += size

            if relu:
                self.res = tf.nn.relu(self.res, name="applying_relu")
                self.flops += size  # we consider relu of requiring 1 computation per number (one max)

            if keep_prob is not None:
                self.res = tf.nn.dropout(self.res, keep_prob=keep_prob, name="applying_dropout")
                # we consider that generating random number count for 1 operation
                self.flops += size  # generate the "size" real random numbers
                self.flops += size  # building the 0-1 vector of size "size" (thresholding "size" random values)
                self.flops += size  # element wise multiplication with res



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
            m_init, v_init = sess.run(tf.nn.moments(tf.matmul(self.input, self.scaled_matrix), [0]))
            # pdb.set_trace()
            sess.run(tf.assign(self.g, scale_init/tf.sqrt(v_init + 1e-10), name="weigth_normalization_init_g"))
            if self.bias:
                sess.run(tf.assign(self.b, -m_init*scale_init/tf.sqrt(v_init + 1e-10), name="weigth_normalization_init_b"))


class EmulDenseLayerwithGD(DenseLayerwithGD):
    def __init__(self, kwardslayer={}, layerClass=None, *args, **kwargs):
        """
        Wrapper to DenseLayerwithGD class, to be used with TensorflowHelpers.ComplexGraph instance, that requires
        the __init__ method to have the "layerClass" argument.
        
        :param kwardslayer: 
        :param layerClass
        :param args: 
        :param kwargs: 
        """
        guided_dropout_mask = None if not "guided_dropout_mask" in kwardslayer else kwardslayer["guided_dropout_mask"]
        guided_dropconnect_mask = None if not "guided_dropconnect_mask" in kwardslayer else kwardslayer["guided_dropconnect_mask"]

        DenseLayerwithGD.__init__(self,*args,
                                   guided_dropout_mask=guided_dropout_mask,
                                   guided_dropconnect_mask=guided_dropconnect_mask,
                                   **kwargs)


class ComplexGraphWithGD(ComplexGraph):
    def __init__(self, data,
                 outputsize,
                 sizes,
                 path, reload,
                 var_x_name={"input"}, var_y_name={"output"},
                 nnType=NNFully, argsNN=(), kwargsNN={},
                 encDecNN=NNFully, args_enc=(), kwargs_enc={},
                 args_dec=(), kwargs_dec={},
                 kwargs_enc_dec=None,
                 spec_encoding={},
                 dropout_spec={}, dropconnect_spec={},
                 has_vae=False,
                 latent_dim_size=None,
                 latent_hidden_layers=(),
                 latent_keep_prob=None,
                 penalty_loss=0.0001,
                 resize_nn=None
                 ):
        """
        This class derived from TensorflowHelpers.ComplexGraph, and implement guided dropout or guided dropconnect
        The parameters are the same as in "TensorflowHelpers.ComplexGraph"
        For recall (may not be up to date, please refer to the official documentation of TensorflowHelpers.ComplexGraph
        for a more complete information)
        
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
        :param path: path of the experiment
        :param reload: should the mask be reloaded or build from scratch
        
        :param has_vae: do you want to add a variationnal auto encoder (between the output of the intermediate neural network and the decoders) 
        :param latent_dim_size: the size of the latent space (int)
        :param latent_hidden_layers: the number of hidden layers of the latent space (ordered iterable of integer)
        :param latent_keep_prob: keep probability for regular dropout for the building of the latent space (affect only the mean)
        :param penalty_loss : penalty to have e and d clsoe to 0 (None for deactivating it)
        """

        modifkwargsNN = copy.deepcopy(kwargsNN)
        dict_kargs = {}
        self._check_everything_ok(modifkwargsNN, dict_kargs)
        self.encgd = {}
        self.masks = {}
        self.penalty_loss = penalty_loss
        # self.penalized = {} # set of tensor penalized by l2 loss

        # deals with guided dropout
        dict_kargs['guided_dropout_mask'] = None
        if len(dropout_spec):
            self._build_guided_dropout(dropout_spec, data, outputsize, path, reload, dict_kargs)

        # deals with guided dropconnect
        if len(dropconnect_spec):
            self._build_guided_dropconnect(dropconnect_spec, data, outputsize, path, reload, dict_kargs)

        modifkwargsNN["kwardslayer"]["kwardslayer"] = dict_kargs
        ComplexGraph.__init__(self,
                              data=data,
                              outputsize=outputsize, sizes=sizes,
                              var_x_name=var_x_name, var_y_name=var_y_name,
                              nnType=nnType, argsNN=argsNN, kwargsNN=modifkwargsNN,
                              encDecNN=encDecNN, args_enc=args_enc, kwargs_enc=kwargs_enc,
                              args_dec=args_dec, kwargs_dec=kwargs_dec, kwargs_enc_dec=kwargs_enc_dec,
                              spec_encoding=spec_encoding,
                              has_vae=has_vae,
                              latent_dim_size=latent_dim_size,
                              latent_hidden_layers=latent_hidden_layers,
                              latent_keep_prob=latent_keep_prob,
                              resize_nn=resize_nn
                              )

    def _logical_or(self, x, y, name="logical_or"):
        """
        compute the logical _or function of two floating tensorflow tensors.
        :param self: 
        :param x: 
        :param y: 
        :param name: an optionnal name for the operation
        :return: 
        """
        return tf.subtract(tf.add(x,y), tf.multiply(x,y), name=name)
    
    def _build_guided_dropout(self, dropout_spec, data, outputsize, path, reload, dict_kargs):
        """
        Build the masks for guided dropout
        :param dropout_spec: 
        :param data: 
        :param outputsize: 
        :param path: 
        :param reload: 
        :param dict_kargs: 
        :return: 
        """
        vars = sorted(list(dropout_spec["vars"]))
        for var in vars:
            if not "proba_select" in dropout_spec:
                msg = "Error: ComplexGraphWithGD: the fields \"proba_select\" in \"dropout_spec\" is not specified"
                raise RuntimeError(msg)
            proba_select = dropout_spec["proba_select"]
            if isinstance(proba_select, type(0.)):
                pb_sel_var = proba_select
            else:
                pb_sel_var = proba_select[var]
            self.encgd[var] = SpecificGDOEncoding(sizeinputonehot=int(data[var].get_shape()[1]),
                                                  sizeout=outputsize,
                                                  proba_select=pb_sel_var,
                                                  path=path,
                                                  reload=reload,
                                                  name="{}_guided_dropout_encoding".format(var))
            self.masks[var] = self.encgd[var](data[var])
            self._update_multiple_mask(dict_kargs, var)

    def _update_multiple_mask(self, dict_kargs, var):
        """
        Combine the mask coming from multiple variable into one, using logical or operator (coded bellow)
        :param var: 
        :return: 
        """
        if dict_kargs['guided_dropout_mask'] is None:
            dict_kargs['guided_dropout_mask'] = self.masks[var]
        else:
            dict_kargs['guided_dropout_mask'] = self._logical_or(dict_kargs['guided_dropout_mask'], self.masks[var])

    def _build_guided_dropconnect(self, dropconnect_spec, data, outputsize, path, reload, dict_kargs):
        """
        Build the masks for guided dropconnect
        Currently mutliple masks are not supported
        :param dropconnect_spec: 
        :param data: 
        :param outputsize: 
        :param path: 
        :param reload: 
        :param dict_kargs: 
        :return: 
        """
        # TODO HEre for multiple variable in guided dropout (logical or and better choice of mask!)
        vars = sorted(list(dropconnect_spec["vars"]))
        var = vars[0]
        if len(vars) > 1:
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
                                         proba_select=proba_select,
                                         path=path,
                                         reload=reload)
        dict_kargs['guided_dropconnect_mask'] = self.encgd(data[var])

    def _check_everything_ok(self, modifkwargsNN, dict_kargs):
        """
        Check some condition, to make sure that guided dropout is applicable.
        
        :param modifkwargsNN: 
        :param dict_kargs: 
        :return: 
        """
        if "layerClass" in modifkwargsNN:
            # TODO nothing to do here!
            msg = "I: ComplexGraphWithGD: you pass a specific class \"{}\" to build the layers."
            msg += " This class should accept the key-word arguments \"guided_dropout_mask\" and \"guided_droconnect_mask\" "
            msg += "in its \"kwardslayer\" constructor"
            print(msg.format(modifkwargsNN["layerClass"]))
        else:
            modifkwargsNN["layerClass"] = EmulDenseLayerwithGD

        if "kwardslayer" in modifkwargsNN:
            if "kwardslayer" in modifkwargsNN["kwardslayer"]:
                dict_kargs = copy.deepcopy(modifkwargsNN["kwardslayer"]["kwardslayer"])
        else:
            modifkwargsNN["kwardslayer"] = {}

        if "guided_dropout_mask" in dict_kargs:
            msg = "W: \"guided_dropout_mask\" in \"kwargsNN\" of object ComplexGraphWithGD will be erased"
            msg += " and replace with the data in \"var_dropout_name\""
            print(msg)
        if "guided_dropconnect_mask" in dict_kargs:
            msg = "W: \"guided_droconnect_mask\" in \"kwargsNN\" of object ComplexGraphWithGD will be erased"
            msg += " and replace with the data in \"var_dropconnect_name\""
            print(msg)

#TODO documentation
class ComplexGraphWithComplexGD(ComplexGraphWithGD):
    def __init__(self, data,
                 outputsize,
                 sizes,
                 path, reload,
                 var_x_name={"input"}, var_y_name={"output"},
                 nnType=NNFully, argsNN=(), kwargsNN={},
                 encDecNN=NNFully, args_enc=(), kwargs_enc={},
                 args_dec=(), kwargs_dec={},
                 kwargs_enc_dec=None,
                 spec_encoding={},
                 masks_spec={},
                 has_vae=False,
                 latent_dim_size=None,
                 latent_hidden_layers=(),
                 latent_keep_prob=None,
                 penalty_loss=0.001,
                 resize_nn=None):
        """
        This class derived from TensorflowHelpers.ComplexGraph, and implement guided dropout or guided dropconnect.
        The parameters are the same as in "TensorflowHelpers.ComplexGraph"
        For recall (may not be up to date, please refer to the officiel documentation of TensorflowHelpers.ComplexGraph for a more complete information)
        
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
        :param path: path of the experiment
        :param reload: should the mask be reloaded or build from scratch
        :param masks_spec: at which layer you apply which mask (basically), and do you use the same mask for all layers
        
        masks_spec is a dictionnary with keys: var, and value a dictionnary with keys:
            "same_mask": True/False (do you want to use the same mask for each layer)
            "layers" : a list of integer (This represent the layers for which this variable will be used as a mask)
            "dropout" : True/False (do you want to use dropout if True, or dropconnect if False)
            "proba_select": the probability for a connection to be selected
            
        :param has_vae: do you want to add a variationnal auto encoder (between the output of the intermediate neural network and the decoders) 
        :param latent_dim_size: the size of the latent space (int)
        :param latent_hidden_layers: the number of hidden layers of the latent space (ordered iterable of integer)
        :param latent_keep_prob: keep probability for regular dropout for the building of the latent space (affect only the mean)
        """

        self.masks_spec = masks_spec
        self.path = path
        self.reload = reload
        self.penalty_loss = penalty_loss

        self.mask_used = {var: False for var in masks_spec.keys()}
        ComplexGraphWithGD.__init__(self,
                                    data=data,
                                    outputsize=outputsize, sizes=sizes,
                                    var_x_name=var_x_name, var_y_name=var_y_name,
                                    nnType=nnType, argsNN=argsNN, kwargsNN=kwargsNN,
                                    encDecNN=encDecNN, args_enc=args_enc, kwargs_enc=kwargs_enc,
                                    args_dec=args_dec, kwargs_dec=kwargs_dec, kwargs_enc_dec=kwargs_enc_dec,
                                    spec_encoding=spec_encoding, path=path,
                                    dropout_spec={}, dropconnect_spec={}, # this won't be use anyway in here, but rather in masks_spec when building the layers
                                    reload=reload,
                                    has_vae=has_vae,
                                    latent_dim_size=latent_dim_size,
                                    latent_hidden_layers=latent_hidden_layers,
                                    latent_keep_prob=latent_keep_prob,
                                    penalty_loss=penalty_loss,
                                    resize_nn=resize_nn)


    def _update_multiple_mask(self, dict_kargs, var):
        """
        For this class, you don't need to do the "logical_or" elementwise for guided dropout.
        Each mask are affected to a different layer.
        :param dict_kargs: 
        :param var: 
        :return: 
        """
        pass

    def _buildintermediateNN(self, nnType, argsNN, input, outputsize, kwargsNN):
        """

        :param nnType:
        :param argsNN: 
        :param input: 
        :param outputsize: 
        :param kwargsNN: 
        :return: 
        """
        layers = []
        if "layersizes" in kwargsNN:
            layers = kwargsNN["layersizes"]
        else:
            msg = "I ComplexGraphWithComplexGD: the key \"layersizes\" is not present in the key-words argument"
            msg += " \"kwargsNN\" specified for building"
            msg += " the intermediate grid layer. We use the first elements of \"argsNN\" instead."
            layers = argsNN[0]

        # build the mask for each variable
        masked_var = {}  # key: layer number, value dictionnary with key:
                         #       "guided_dropout_mask": the mask for guided dropout
                         #       "guided_dropconnect_mask" :  the mask for guided dropconnect
        tensors_gdo = {}
        tensors_gdc = {}
        self._build_mask_var(tensors_gdo, tensors_gdc, outputsize, masked_var, layers)

        # forward the builts kwardlayers to nnType constructor, in order to build the layer
        kwardslayers = []  #copy.deepcopy(kwardslayer["kwardslayer"])
        for i in range(len(layers)):
            tmp_kwl = copy.deepcopy(kwargsNN["kwardslayer"])
            if not "kwardslayer" in tmp_kwl:
                tmp_kwl["kwardslayer"] = {}
            if i in masked_var:
                tmp_kwl["kwardslayer"]["guided_dropout_mask"] = masked_var[i]["guided_dropout_mask"]
                tmp_kwl["kwardslayer"]["guided_dropconnect_mask"] = masked_var[i]["guided_dropconnect_mask"]
            kwardslayers.append(tmp_kwl)

        kwargsNN["kwardslayer"] = kwardslayers
        # pdb.set_trace()
        self.nn = nnType(*argsNN,
                         input=input,
                         outputsize=outputsize,
                         **kwargsNN)

        # warnings to prevent misusage ot the masks
        nblayer = len(layers)
        for var, layerspec in self.masks_spec.items():
            if max(layerspec["layers"]) >= nblayer:
                msg = "W ComplexGraphWithComplexGD: the variable {} is set to be used at layers {}."
                msg += " But the network counts only {} layers. "
                msg += "The masks won't be set for layers {}."
                print(msg.format(var, layerspec["layers"], nblayer, [el for el in layerspec["layers"] if el >= nblayer]))

        # for var, used in self.mask_used.items():
        #     if not used:
        #         msg = "W ComplexGraphWithComplexGD: the variable {} will not be used in any mask for guided dropout!"
        #         msg += " You ask it to be used first in layer {}, but the network counts only {} layers"
        #         print(msg.format(var, min(self.masks_spec[var]["layers"]), nblayer))

    def _build_mask_var(self, tensors_gdo, tensors_gdc, outputsize, masked_var, layers):
        """
        
        :param tensors_gdo: 
        :param tensors_gdc: 
        :param outputsize: 
        :param masked_var: 
        :param layers: 
        :return: 
        """
        for var, layerspec in self.masks_spec.items():
            initmask = True
            tensors_gdo[var] = {}
            tensors_gdc[var] = {}
            self.encgd[var] = {}
            firstlayerthisvar = -1
            for layernum in layerspec["layers"]:
                if initmask or (not layerspec["same_mask"]):
                    # define the mask
                    proba_select = layerspec["proba_select"] if "proba_select" in layerspec else None
                    nbconnections = layerspec["nbconnections"] if "nbconnections" in layerspec else None
                    keep_prob = layerspec["keep_prob"] if "keep_prob" in layerspec else None
                    var_name = var
                    if not layerspec["same_mask"]:
                        var_name = var + "_"+str(layernum)
                    if layerspec["guided_dropout"]:
                        enco = SpecificGDOEncoding(sizeinputonehot=int(self.data[var].get_shape()[1]),
                                                   sizeout=outputsize,
                                                   proba_select=proba_select,
                                                   nbconnections=nbconnections,
                                                   path=self.path,
                                                   reload=self.reload,
                                                   name="{}_guided_dropout_encoding".format(var_name),
                                                   keep_prob=keep_prob)
                        tensors_gdo[var][layernum] = enco(self.data[var])
                        tensors_gdc[var][layernum] = None
                    else:
                        nrow = layerspec["nrow"]
                        enco = SpecificGDCEncoding(sizeinputonehot=int(self.data[var].get_shape()[1]),
                                                   proba_select=proba_select,
                                                   nbconnections=nbconnections,
                                                   path=self.path,
                                                   reload=self.reload,
                                                   name="{}_guided_droconnect_encoding".format(var_name),
                                                   nrow=nrow,
                                                   ncol=outputsize,
                                                   keep_prob=keep_prob)
                        tensors_gdo[var][layernum] = None
                        tensors_gdc[var][layernum] = enco(self.data[var])
                    self.encgd[var][layernum] = enco
                    firstlayerthisvar = layernum
                    initmask = False
                elif layerspec["same_mask"]:
                    # or reuse previously defined masks
                    tensors_gdo[var][layernum] = tensors_gdo[var][firstlayerthisvar]
                    tensors_gdc[var][layernum] = tensors_gdc[var][firstlayerthisvar]

                if not layernum in masked_var:
                    masked_var[layernum] = {}
                    masked_var[layernum]["guided_dropout_mask"] = tensors_gdo[var][layernum]
                    masked_var[layernum]["guided_dropconnect_mask"] = tensors_gdc[var][layernum]
                else:
                    # TODO logical or in that case!
                    msg = "I ComplexGraphWithComplexGD: for now multiple mask acting on one layer is not supported. "
                    msg += " The layer {} was present in {} and at least another variable."
                    print(msg.format(layernum, var))

                if layernum >= len(layers):
                    msg = "W ComplexGraphWithComplexGD you define a mask for layer {}, but your networks counts "
                    msg += " only {} layers. To mute this warning, either reset the arguments \"layers\" in \"kwargsNN\" "
                    msg += " when building the key-word argument of layers (should be in \"graphkwargs\") "
                    msg += " or the \"layers\" argument passed to variable {} of \"masks_spec\" argument of ComplexGraphWithComplexGD."
                    print(msg.format(layernum, var))

    def init_loss(self, loss):
        """
        Assign the loss
        :param loss: the loss tensor use for training (reconstruction loss) : I need to add the KL-divergence loss if vae is used
                    this tensor must be in tf.float32 type
        :return:
        """
        self.loss = loss
        if self.penalty_loss is not None:
            for var, layerspec in self.masks_spec.items():
                for layernum, gd_val in self.encgd[var].items():
                    nbconnections = layerspec["nbconnections"] if "nbconnections" in layerspec else 1
                    # layerGD = self.encgd[var][layernum]
                    lsi_block = self.nn.layers[layernum]

                    loss = tf.add(loss, self.penalty_loss * tf.reduce_sum(tf.cast(self.data[var], tf.float32)) / nbconnections * (
                        tf.nn.l2_loss(lsi_block.w_e_fp32) + tf.nn.l2_loss(lsi_block.w_d_fp32)),
                        name="adding_penalty_{}_{}".format(var, layernum)
                    )
        self.loss = loss
        # pdb.set_trace()
        return loss

from TensorflowHelpers import DenseLayer

class LeapLayer:
    def __init__(self, input, latent_dim_size, resize_nn,
                 nnTypeE, argsE, kwargsE,
                 nnTypeD, argsD, kwargsD,
                 size_out,
                 size_L, n_layers_L,
                 latent_jump, tau):
        self.enc_output_raw = input
        self.size_out = size_out
        # self.data = data
        self.tau = tau
        # 3. build the neural network
        with tf.variable_scope("E"):
            self.nn = None
            if resize_nn is not None:
                self.resize_layer = DenseLayer(input=self.enc_output_raw, size=latent_dim_size,
                                               relu=False, bias=False,
                                               weight_normalization=False,
                                               keep_prob=None, layernum="resizing_layer")
                self.enc_output = self.resize_layer.res
            else:
                self.resize_layer = None
                self.enc_output = self.enc_output_raw

            self.h_x = None
            self.E, self.h_x = self._build(nnType=nnTypeE, args=argsE,
                                   input=self.enc_output,
                                   outputsize=latent_dim_size,
                                   kwargs=kwargsE)
        self.my_flop = 0
        self.my_nb_params = 0

        # make the latent leap
        with tf.variable_scope("latent_leap" if latent_jump else "residual_path"):
            # var = var_leap[0]
            # size_var = int(self.data[var].get_shape()[1])*nb_axis_per_dim_of_tau

            # old leap net with e and d
            # ## build e
            # self.h_tau_e = None
            # with tf.variable_scope("e"):
            #     self.e, self.h_tau_e = self._build(nnType=nnType_e, args=args_e,
            #                                input=self.h_x, outputsize=size_var, kwargs=kwargs_e)
            #
            # ## at this stage tau is ready to be projected on the axis
            # if latent_jump:
            #
            #     with tf.variable_scope("projection"):
            #         enco = SpecificGDOEncoding(sizeinputonehot=int(self.data[var].get_shape()[1]),
            #                                    sizeout=outputsize,
            #                                    nbconnections=nb_axis_per_dim_of_tau,
            #                                    path=self.path,
            #                                    reload=self.reload,
            #                                    name="{}_guided_dropout_encoding".format(var),
            #                                    keep_prob=None)
            #         self.enc_var_gdo = enco(self.data[var])
            #         self.h_tau_after_proj = tf.multiply(self.h_tau_e, self.enc_var_gdo)
            # else:
            #     self.h_tau_after_proj = self.h_tau_e
            #
            # ## build d
            # self.h_tau = None
            # with tf.variable_scope("d"):
            #     self.d, self.h_tau = self._build(nnType=nnType_d, args=args_d,
            #                  input=self.h_tau_after_proj, outputsize=latent_dim_size, kwargs=kwargs_d)


            # new leap net, Balthazar style
            s_tau = self.tau.shape[-1]
            D = self.h_x.shape[-1]

            h = tf.reshape(self.h_x, [-1, 1, 1, D]) # assign shape [Nbatch, 1, 1, D]
            h = h * tf.ones([1, s_tau, 1, 1]) # copy "size of tau" times [Nbatch, |tau|, 1, D]
            self.my_flop += s_tau * D # copy

            input_dim = D
            output_dim = size_L
            name = "leap"
            non_lin = tf.nn.relu
            self.wbs = []
            for layer in range(n_layers_L):

                # For the first and last layers, specify the right dimension
                # latent_dimension_left = d
                # latent_dimension_right = d
                # if layer == 0:
                #     latent_dimension_left = d_in
                if layer == n_layers_L - 1:
                    output_dim = D

                w_fp32 = tf.get_variable(name= "w_{}_{}".format(name, layer+1),#'w_' + name + '_' + str(layer + 1),
                                    shape=[1, s_tau, input_dim, output_dim],
                                    initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32,
                                                                                     uniform=False),
                                    trainable=True,
                                    dtype=tf.float32)
                self.my_nb_params += s_tau * input_dim * output_dim
                b_fp32 = tf.get_variable(name='b_{}_{}'.format(name, layer+1),
                                    shape=[1, s_tau, 1, output_dim],
                                    initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32,
                                                                                     uniform=False),
                                    trainable=True,
                                    dtype=tf.float32)
                if DTYPE_USED != tf.float32:
                    w = tf.cast(w_fp32, DTYPE_USED)
                    b = tf.cast(b_fp32, DTYPE_USED)
                else:
                    w = w_fp32
                    b = b_fp32

                self.wbs.append((w,b))

                self.my_nb_params += s_tau * output_dim


                w = w * tf.ones([tf.shape(h)[0], 1, 1, 1])
                self.my_flop += tf.shape(h)[0] * s_tau * input_dim * output_dim  # copy

                h = non_lin(tf.matmul(h, w, name="build_L_layer_{}".format(layer)) + b)
                self.my_flop += s_tau*(2*input_dim*output_dim-output_dim) # matrix multiplication
                self.my_flop += s_tau * output_dim  # adding bias
                self.my_flop += tf.shape(h)[0] * s_tau * output_dim  # non linearity

                input_dim = size_L

            h = tf.transpose(h, [0, 3, 2, 1])

            tau = tf.reshape(self.tau, [-1, 1, s_tau, 1])  # reshape [Nbatch, 1, |tau|, 1]
            tau = tau * tf.ones([1, D, 1, 1])   # copy "size of tau" times [Nbatch, D, |tau|, 1]
            self.my_flop += s_tau * D  # non linearity

            h = tf.matmul(h, tau, name="tau_element_wise_sum")  # results of size [Nbatch, D, 1, 1]
            self.my_flop += D*(s_tau) # matrix multiplication

            self.leap = tf.reshape(h, [-1, D])  # reshape [Nbatch, D] = original size

        ## really make the leap now:
        self.h_after_leap = self.h_x + self.leap

        output_D_size = size_out  #latent_dim_size if kwargs_postproc is not None else self.size_out
        with tf.variable_scope("D"):
            self.D, self.hat_y_tmp = self._build(nnType=nnTypeD, args=argsD,
                                     input=self.h_after_leap,
                                     outputsize=output_D_size,
                                     kwargs=kwargsD)

    def _build(self, nnType, args, input, outputsize, kwargs):
        """

                :param nnType:
                :param argsNN:
                :param input:
                :param outputsize:
                :param kwargsNN:
                :return:
                """
        try:
            nn = nnType(*args,
                        input=input,
                        outputsize=outputsize,
                        **kwargs)
        except Exception as except_:
            print(except_)
            pdb.set_trace()
            raise
        return nn, nn.pred

    def initwn(self, sess):
        raise NotImplementedError()

    def getnbparam(self):
        """
        :return:  the number of total free parameters of the neural network"""
        res = self.resize_layer.nbparams if self.resize_layer is not None else 0
        res += self.E.getnbparam()
        res += self.my_nb_params
        res += self.D.getnbparam()
        return res

    def getflop(self):
        """
        flops are computed using formulas in https://mediatum.ub.tum.de/doc/625604/625604
        it takes into account both multiplication and addition.
        Results are given for a minibatch of 1 example for a single forward pass.
        :return: the number of flops of the neural network build
        """
        res = self.resize_layer.flops if self.resize_layer is not None else 0
        res += self.E.getflop()
        res += self.my_flop
        res += self.D.getflop()
        return res

class LeapcVAE:
    def __init__(self, data,
                 outputsize,
                 sizes,
                 path, reload,
                 latent_dim_size,
                 leap=True,
                 var_tau=("tau", ),
                 # resnet_if_no_jump=True,
                 # variable as input and output
                 var_x_name=("input",), var_y_name=("output",),

                 # pre processing
                 NN_preproc=NNFully,
                 args_preproc=(), kwargs_preproc={},
                 resize_nn=True,

                 # latent leaps (no sharing between dimension) USED ONLY WHEN leap IS TRUE
                 # encoder E
                 nnTypeE=NNFully, argsE=(), kwargsE={},
                 # decoder D
                 nnTypeD=NNFully, argsD=(), kwargsD={},
                 # leap L
                 n_layers_L=1,size_L=10,
                 latent_dim_size_leap=20,

                 # standard NN USED ONLY WHEN leap is FALSE
                 layersizes=(10,),

                 # post processing
                 NN_postproc=NNFully,
                 args_postproc=(), kwargs_postproc={},

                 ):
        """
        """
        spec_encoding = {}  # specific pre processing of the data, currently not used

        if len(var_tau) > 1:
            raise RuntimeError("For now you can only make a latent leap with one variable")

        self.data = data  # the dictionnary of data pre-processed as produced by an ExpData instance
        self.outputname = var_y_name  # name of the output variable, should be one of the key of self.data
        self.inputname = var_x_name
        self.path = path
        self.reload = reload
        # self.vars_out = var_y_name

        # dictionnary of "ground truth" data
        self.true_dataY = {k: self.data[k] for k in self.outputname}
        self.true_dataX = {k: self.data[k] for k in self.inputname}
        self.size_in = 0
        for _, v in self.true_dataX.items():
            self.size_in += int(v.get_shape()[1])

        self.size_out = 0
        for _, v in self.true_dataY.items():
            self.size_out += int(v.get_shape()[1])

        # Preprocess independantly each data types (initialization procedure)
        with tf.variable_scope("preprocessing"):
            if kwargs_preproc is None:
                """
                In this case, there will not be any encoders, all the input data will be concatenated
                """
                # 1. build the input layer
                self.dimin = {}  # to memorize which data goes where
                self.has_preproc = False
                prev = 0
                tup = tuple()
                for el in sorted(self.inputname):
                    if el in spec_encoding:
                        tup += (spec_encoding[el](self.data[el]),)
                    else:
                        tup += (self.data[el],)
                    this_size = int(tup[-1].get_shape()[1])
                    self.dimin[el] = (prev, prev + this_size)
                    prev += this_size
                self.input = tf.concat(tup, axis=1, name="input_concatenantion")
            else:
                self.dimin = None
                self.has_preproc = True

            # 1. build the preprocessing NN
            self.output_preproc = {}
            self.pre_proc = {}
            if self.has_preproc:
                self._buildencoders(sizes, spec_encoding, NN_preproc, args_preproc, kwargs_preproc)

                # self.input = tf.zeros(shape=(None, 0), dtype=DTYPE_USED)
                tup = tuple()
                for el in sorted(self.inputname):
                    tup += (self.output_preproc[el],)
                self.enc_output_raw = tf.concat(tup, axis=1, name="encoder_output_concatenantion")
            else:
                self.enc_output_raw = self.input

        self.tau = tf.concat([self.data[var] for var in var_tau], axis=1, name="tau_concatenantion")
        with tf.variable_scope("cVAE"):
            with tf.variable_scope("encoder"):
                if leap:
                    self.Enc = LeapLayer(input=self.enc_output_raw, latent_dim_size=latent_dim_size_leap, resize_nn=resize_nn,
                    nnTypeE=nnTypeE, argsE=argsE, kwargsE=kwargsE, nnTypeD=nnTypeD, argsD=argsD, kwargsD=kwargsD,
                                       size_out=2*latent_dim_size,
                                       size_L=size_L, n_layers_L=n_layers_L, tau=self.tau,
                    latent_jump=leap,)
                    self.h = self.Enc.hat_y_tmp
                else:
                    self.Enc = NNFully(input=tf.concat((self.enc_output_raw, self.tau), axis=1),
                                      outputsize=2*latent_dim_size,
                                      layersizes=layersizes
                                      )
                    self.h = self.Enc.pred

            with tf.variable_scope("reparam_trick"):
                self.mu = self.h[:, :latent_dim_size]
                self.logvar = self.h[:, latent_dim_size:]
                self.eps = tf.random_normal(shape=(latent_dim_size,))
                self.z = self.mu + tf.exp(self.logvar / 2) * self.eps

            with tf.variable_scope("decoder"):
                if leap:
                    self.Dec = LeapLayer(input=self.z, latent_dim_size=latent_dim_size_leap, resize_nn=resize_nn,
                    nnTypeE=nnTypeE, argsE=argsE, kwargsE=kwargsE, nnTypeD=nnTypeD, argsD=argsD, kwargsD=kwargsD,
                                       size_out=latent_dim_size_leap if kwargs_postproc is not None else self.size_out,
                                       size_L=size_L, n_layers_L=n_layers_L, tau=self.tau,
                    latent_jump=leap,)
                    self.hat_y_tmp = self.Dec.hat_y_tmp
                else:
                    self.Dec = NNFully(input=tf.concat((self.z, self.tau), axis=1),
                                      outputsize=latent_dim_size_leap if kwargs_postproc is not None else self.size_out,
                                      layersizes=layersizes
                                      )
                    self.hat_y_tmp = self.Dec.pred

                # self.hat_y_tmp = self.Dec.hat_y_tmp
        # post processing independantly each data types (initialization procedure)
        with tf.variable_scope("post_processing"):
            if kwargs_postproc is None:
                """
                In this case, there will not be any decoders, all the output data will be concatenated
                """
                # 2. build the output layer
                self.dimout = {}  # to memorize which data goes where
                self.has_postproc = False
                prev = 0
                tup = tuple()
                for el in sorted(self.outputname):
                    tup += (self.data[el],)
                    this_size = int(self.data[el].get_shape()[1])
                    self.dimout[el] = (prev, prev + this_size)
                    prev += this_size
                self.output = tf.concat(tup, axis=1, name="output_concatenantion")
            else:
                self.dimout = None
                self.has_postproc = True

            self.y_hat = {}
            self.post_proc = {}
            if self.has_postproc:
                self._builddecoders(NN_postproc, args_postproc, kwargs_postproc, self.hat_y_tmp)
            else:
                self.y_hat = {}  # dictionnary of output of the NN
                for varn in sorted(self.outputname):
                    be, en = self.dimout[varn]
                    self.y_hat[varn] = self.hat_y_tmp[:, be:en]


        # 7. create the fields summary and loss that will be created in ExpModel and assign via "self.init"
        self.mergedsummaryvar = None
        self.loss = None
        self.vars_out = self.y_hat
        with tf.name_scope('kl_loss'):
            self.kl_loss = 0.5 * tf.reduce_sum(tf.exp(self.logvar) + self.mu ** 2 - 1. - self.logvar, 1)

    def _build(self, nnType, args, input, outputsize, kwargs):
        """

                :param nnType:
                :param argsNN:
                :param input:
                :param outputsize:
                :param kwargsNN:
                :return:
                """
        try:
            nn = nnType(*args,
                        input=input,
                        outputsize=outputsize,
                        **kwargs)
        except Exception as except_:
            print(except_)
            pdb.set_trace()
            raise
        return nn, nn.pred

    def initwn(self, sess):
        """
        Initialize the weights for weight normalization
        :param sess: a tensorflow session
        :return:
        """
        for _, v in self.pre_proc.items():
            v.initwn(sess=sess)
        self.Enc.initwn(sess=sess)
        self.Dec.initwn(sess=sess)
        for _, v in self.post_proc.items():
            v.initwn(sess=sess)

    def get_true_output_dict(self):
        """
        :return: the output data dictionnary. key: varname, value=true value of this data
        """
        return self.true_dataY

    def get_input_size(self):
        """
        :return: the number of columns (variables) in input
        """
        return self.size_in

    def get_output_size(self):
        """
        :return: the number of columns (variables) in output
        """
        return self.size_out

    def getnbparam(self):
        """
        :return:  the number of total free parameters of the neural network"""
        #TODO
        res = 0
        for _, v in self.pre_proc.items():
            res += v.getnbparam()
        self.Enc.getnbparam()
        self.Dec.getnbparam()
        for _, v in self.post_proc.items():
            res += v.getnbparam()
        return res

    def getflop(self):
        """
        flops are computed using formulas in https://mediatum.ub.tum.de/doc/625604/625604
        it takes into account both multiplication and addition.
        Results are given for a minibatch of 1 example for a single forward pass.
        :return: the number of flops of the neural network build
        """
        res = 0
        for _, v in self.pre_proc.items():
            res += v.getflop()
        self.Enc.getflop()
        self.Dec.getflop()
        for _, v in self.post_proc.items():
            res += v.getflop()
        return res

    def _buildencoders(self, sizes, spec_encoding, encDecNN, args_enc, kwargs_enc):
        """
        Build the encoder networks
        :param sizes:
        :param spec_encoding:
        :param encDecNN:
        :param args_enc:
        :param kwargs_enc:
        :return:
        """
        with tf.variable_scope("preprocessing"):
            for varname in sorted(self.inputname):
                with tf.variable_scope(varname):
                    if not varname in sizes:
                        msg = "ComplexGraph._buildencoders the variable {} is not in \"sizes\" argument but in \"var_x_name\""
                        msg += " (or \"var_y_name\")"
                        raise RuntimeError(msg.format(varname))
                    size_out = sizes[varname]
                    if varname in spec_encoding:
                        input_tmp = spec_encoding[varname](self.data[varname])
                    else:
                        input_tmp = self.data[varname]
                    tmp = encDecNN(*args_enc,
                                   input=input_tmp,
                                   outputsize=size_out,
                                   **kwargs_enc)
                    self.pre_proc[varname] = tmp
                    # self.post_proc[varname] = tmp.pred

    def _builddecoders(self, encDecNN, args_dec, kwargs_dec, inputdec):
        """
        Build the decoder networks
        :param sizes:
        :param encDecNN:
        :param args_dec:
        :param kwargs_dec:
        :return:
        """
        with tf.variable_scope("postprocessing"):
            for varname in sorted(self.outputname):
                with tf.variable_scope(varname):
                    # size_out = sizes[varname]
                    tmp = encDecNN(*args_dec,
                                   input=inputdec,
                                   outputsize=int(self.data[varname].get_shape()[1]),
                                   **kwargs_dec)
                    self.post_proc[varname] = tmp
                    self.y_hat[varname] = tmp.pred
                    # self.size_out += int(tmp.pred.get_shape()[1])

    def _have_latent_space(self):
        return False

    def init_loss(self, loss):
        """
        Assign the loss
        :param loss: the loss tensor use for training (reconstruction loss) : I need to add the KL-divergence loss if vae is used
        :return:
        """
        # pdb.set_trace()
        self.loss = loss

        return self.loss + self.kl_loss

    def init_summary(self, mergedsummaryvar):
        """
        Assign the summary 'mergedsummaryvar' for easier access
        :param mergedsummaryvar: the summary of everything to be save by tensorboard
        :return:
        """
        self.mergedsummaryvar = mergedsummaryvar

    def startexp(self, sess):
        """
        TODO documentation
        :param sess:
        :return:
        """
        pass

    def tell_epoch(self, sess, epochnum):
        """
        TODO documentation
        :return:
        """
        pass

    def getoutput(self):
        """
        :return: a dictionnray corresponding to the output variables. keys; variables names, values: the tensor of the forward pass
        """
        # pdb.set_trace()
        return self.y_hat

    def run(self, sess, toberun):
        """
        Use the tensorflow session 'sess' to run the graph node 'toberun' with data 'data'
        :param sess: a tensorflow session
        :param toberun: a node in the tensorflow computation graph to be run...
        :return:
        """
        return sess.run(toberun)