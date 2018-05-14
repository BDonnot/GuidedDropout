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
        self.bias = False
        self.res = None
        # guided_dropconnect_mask = kwardslayer["guided_dropconnect_mask"]
        # guided_dropout_mask = kwardslayer["guided_dropout_mask"]

        with tf.variable_scope("dense_layer_{}".format(layernum)):
            self.w_e = tf.get_variable(name="weights_matrix_enc",
                                      shape=[nin_, size],
                                      initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
                                      # initializer=tf.get_default_graph().get_tensor_by_name(tf.get_variable_scope().name+"/weights_matrix:0"),
                                      trainable=True)
            self.nbparams += int(nin_ * size)
            self.w_d = tf.get_variable(name="weights_matrix_dec",
                                      shape=[size, nin_],
                                      initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
                                      # initializer=tf.get_default_graph().get_tensor_by_name(tf.get_variable_scope().name+"/weights_matrix:0"),
                                      trainable=True)
            self.nbparams += int(nin_ * size)

            if weight_normalization:
                raise RuntimeError("weight normalization not implemented yet")
                self.weightnormed = True
                self.g = tf.get_variable(shape=[size],
                                         name="weight_normalization_g",
                                         initializer=tf.constant_initializer(value=0.0, dtype="float32"),
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
                self.w = self.w_e

            if guided_dropconnect_mask is not None:
                raise RuntimeError("guided dropconnect not implemented yet")
                self.w = tf.multiply(self.w, guided_dropconnect_mask, name="applying_guided_dropconnect")
                self.flops += nin_*size

            res = tf.matmul(self.input, self.w, name="multiplying_weight_matrix")
            self.flops += 2 * nin_ * size - size

            if bias:
                self.bias = True
                self.b = tf.get_variable(shape=[size],
                                         initializer=tf.constant_initializer(value=1.0, dtype="float32"),
                                         # initializer=tf.get_default_graph().get_tensor_by_name(tf.get_variable_scope().name+"/bias:0"),
                                         name="bias",
                                         trainable=True)
                self.nbparams += int(size)
                res = tf.add(res, self.b, name="adding_bias")
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
                 penalty_loss=0.0001
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
                              latent_keep_prob=latent_keep_prob
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
                 penalty_loss=0.001):
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
                                    penalty_loss=penalty_loss)


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
        :return:
        """
        self.loss = loss
        if self.penalty_loss is not None:
            for var, layerspec in self.masks_spec.items():
                for layernum, gd_val in self.encgd[var].items():
                    nbconnections = layerspec["nbconnections"] if "nbconnections" in layerspec else 1
                    layerGD = self.encgd[var][layernum]
                    lsi_block = self.nn.layers[layernum]

                    loss = tf.add(loss, self.penalty_loss * tf.reduce_sum(self.data[var], axis=1) / nbconnections * (
                        tf.nn.l2_loss(lsi_block.w_e) + tf.nn.l2_loss(lsi_block.w_d)),
                        name="adding_penalty_{}_{}".format(var, layernum)
                    )
        self.loss = loss
        # pdb.set_trace()
        return loss