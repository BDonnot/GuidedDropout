import os
import copy
import re

import numpy as np
import tensorflow as tf

import pdb



class EncodingRaw:
    def __init__(self, num_elem, nullElemenEnc):
        """Encode one dimension only, for onehot only
        :param num_elem: the number of element in the variable encoded
        :param nullElemenEnc: the encoding of the null element
        """
        self.enco = {}
        self.nullElemenEnc = nullElemenEnc
        self.size = nullElemenEnc.shape
        self.num_elem = num_elem
        self.nullelemenArr = np.zeros(num_elem, dtype=np.int)
        self.nullElemenkey = self.nullelemenArr.tostring()
        self.enco[self.nullElemenkey] = copy.deepcopy(nullElemenEnc)
        self.num = len(self.nullElemenkey)

    def __getitem__(self, item):
        """
        Return the mask associated with the item given in input
        :param item: 
        :return: 
        """
        if not item in self.enco.keys():
            if len(item) != self.num:
                raise RuntimeError(
                    "EncodingRaw.__getitem__: unsuported querying for object of size {} (normal size should be {})".format(
                        len(item), self.num))
            # tmp = self.nullelemenArr
            res = self.nullElemenEnc
            arr_tmp = np.fromstring(item, dtype=np.int)
            for id_el, el in enumerate(arr_tmp):
                if el != 0:
                    tmp_ = copy.copy(self.nullelemenArr)
                    tmp_[id_el] = el
                    tmp_ = self.__getitem__(tmp_.tostring())
                    res = np.logical_or(res, tmp_)
            self.enco[item] = res.reshape(self.size).astype(np.float32)
        return self.enco[item]

    def __setitem__(self, key, value):
        self.enco[key] = value

    def keys(self):
        return self.enco.keys()

    def save(self, path, name):
        """
        save the masks for every key
        :param path: the path where the data should be stored
        :param name: the name used to save (all mask will be stored in path/name/...npy)
        :return: 
        """
        mypath = os.path.join(path, name)
        if not os.path.exists(mypath):
            os.mkdir(os.path.join(mypath))
        for key, mask in self.enco.items():
            arr = np.fromstring(key, dtype=np.int)
            nm = []
            for i, v in enumerate(arr):
                if v != 0:
                    nm.append("{}".format(i))
            if len(nm) == 0:
                nm = "bc"
            else:
                nm = "_".join(nm)
            nm += ".npy"
            np.save(file=os.path.join(mypath, nm), arr=mask)

    def reload(self, path, name):
        """
        reload the mask for every keys
        :param path: the path from where the data should be restored
        :param name: 
        :return: 
        """
        # pdb.set_trace()
        mypath = os.path.join(path, name)
        if not os.path.exists(mypath):
            msg = "E: EncodingRaw.reload: the directory {} doest not exists".format(mypath)
            raise RuntimeError(msg)
        del self.enco
        self.enco = {}
        # self.nullelemenArr = np.zeros(num_elem, dtype=np.int)
        # self.nullElemenkey = self.nullelemenArr.tostring()

        if not os.path.exists(os.path.join(mypath, "bc.npy")):
            msg = "E: EncodingRaw.reload: the file in bc.npy is nto located at {}, but it should".format(mypath)
            raise RuntimeError(msg)

        arr = np.load(os.path.join(mypath, "bc.npy"))
        self.enco[self.nullElemenkey] = copy.deepcopy(arr)
        self.nullElemenEnc = copy.deepcopy(arr)
        for fn in os.listdir(mypath):
            if not self._isfnOK(fn):
                continue
            key = self._extractkey(fn)
            self.enco[key] = np.load(os.path.join(mypath, fn))

    def _isfnOK(self, fn):
        """
        return true if the file is a 'mask' file
        :param fn: 
        :return: 
        """
        return re.match("([0-9]+(\_[0-9]+)*)\.npy", fn) is not None

    def _extractkey(self, fn):
        """
        Extract the key to the stored array
        :param fn: 
        :return: 
        """
        fn = re.sub("\.npy", "", fn)
        fns = fn.split("_")
        arr = np.zeros(self.num_elem, dtype=np.int)
        for el in fns:
            arr[int(el)] = 1
        # print("fn: {}\n key: {}".format(fn, arr))
        return arr.tostring()


class SpecificGDCEncoding:
    def __init__(self, sizeinputonehot, nrow, ncol,
                 proba_select=0.25, name="guided_dropconnect_encoding",
                 path=".", reload=False):
        """
        Encoding for guided dropconnect
        :param nrow
        :param ncol
        :param sizeinputonehot: size of the input vector
        :param probaselect: the probability for a given connection to be selected as a mask
        :param path: path where mask will be stored
        :param reaload: do you want to reload (T) or build (F) the mask
        """
        self.size_in = nrow
        self.size_out = ncol
        self.proba_select = proba_select
        self.sizeinputonehot = sizeinputonehot  # for now works only with one-hot data
        choices = self.which_goes_where()
        # build the null element encoding:
        nullElemenEnc = np.zeros(shape=self.proper_size_init(), dtype=np.float32)
        nullElemenEnc[choices == self.sizeinputonehot] = 1.
        self.masks = EncodingRaw(num_elem=self.sizeinputonehot, nullElemenEnc=nullElemenEnc)
        self.name_op = name
        if not reload:
            null_key = np.zeros(self.sizeinputonehot, dtype=np.int)
            # build the other masks
            for i in range(self.sizeinputonehot):
                tmp_key = copy.deepcopy(null_key)
                tmp_key[i] = 1
                tmp_val = copy.deepcopy(nullElemenEnc)
                tmp_val[choices == i] = 1
                self.masks[tmp_key.tostring()] = tmp_val
                # print(tmp_key.tostring())
            self.masks.save(path=path, name=name)
        else:
            self.masks.reload(path=path, name=name)

    def __call__(self, x):
        """
        Convert the input 'x' in the associated guided dropconnect mask
        :param x: a tensorflow tensor: the input
        :return: the associated mask
        """
        res = tf.py_func(func=self.guided_drop, inp=[x], Tout=tf.float32, name=self.name_op)
        res.set_shape(self.proper_size_tf())
        return res

    def guided_drop(self, x):
        """
        Function that implement guided dropconnect: all element should have the same "tau" values
        :param x: a numpy two dimensional array 
        :return: 
        """
        # pdb.set_trace()
        test_same_vect = np.apply_along_axis(lambda x: x.tostring(), 1, x)
        if len(np.unique(test_same_vect)) != 1:
            msg = "guided_dropconnect : different vector use for getting the mask. This is for now unsupported."
            raise RuntimeError(msg)
        x = x[0, :]  # only the first line is relevant here, the other should be equal
        # todo check stuff before here!
        if x.shape != (self.sizeinputonehot,):
            msg = "guided_dropconnect: you give a vector with size {}, masks are defined with input of size {}".format(
                x.shape[0], self.sizeinputonehot)
            raise RuntimeError(msg)
        x = x.reshape(self.sizeinputonehot)
        x = x.astype(np.int)
        xhash = x.tostring()
        return self.masks[xhash]

    def which_goes_where(self):
        """
        Decides at random which connection is assign to which masks.
        :return: a vector of the proper shape, with integer between 0 and "ndim*" stating which connection is assigned 
        :return: to which mask. (connections numbered ndim are assign to the "null element" encoding)
        """

        ## try to equilibrate the number of connection per dimension
        numberofconnections = self.size_in * self.size_out
        nb_neutral = int((1 - self.proba_select) * numberofconnections)
        rest_to_fill = numberofconnections - nb_neutral
        distributed = list(range(self.sizeinputonehot))
        # pdb.set_trace()
        if rest_to_fill // self.sizeinputonehot == 0:
            msg = "W /!\ guided_dropout / dropconnect: There are 0 connections assigned to some masks.\n"
            msg += "W /!\ Masking will not work properly!\n"
            msg += "W /!\ Consider adding more units to the masks or increasing 'proba_select'.\n"
            print(msg)
        choices = distributed * (rest_to_fill // self.sizeinputonehot)
        choices += distributed[:(rest_to_fill % self.sizeinputonehot)]
        choices += [self.sizeinputonehot] * nb_neutral
        choices = np.array(choices)
        for i in range(self.size_out):
            # make sur the shuffling shuffles properly
            np.random.shuffle(choices)
        choices = choices.reshape(self.proper_size_init())
        return choices

    def proper_size_tf(self):
        return (self.size_in, self.size_out)

    def proper_size_init(self):
        return (self.size_in, self.size_out)


class SpecificGDOEncoding(SpecificGDCEncoding):
    def __init__(self, sizeinputonehot, sizeout, proba_select=0.25,
                 name="guided_dropout_encoding",
                 path=".", reload=False):
        """

        :param sizeout: 
        :param proba_select: 
        """
        SpecificGDCEncoding.__init__(self, sizeinputonehot=sizeinputonehot, nrow=1,
                                     ncol=sizeout, proba_select=proba_select, name=name,
                                     path=path, reload=reload)

    def proper_size_init(self):
        return self.size_out

    def proper_size_tf(self):
        return tf.Dimension(None), self.size_out

    def guided_drop(self, x):
        """
        Function that implement guided dropout: each element can have a different tau
        :param x: a numpy two dimensional array 
        :return: 
        """
        res = np.apply_along_axis(func1d=lambda x: self.guided_aux(x), axis=1, arr=x)
        return res

    def guided_aux(self, x):
        # pdb.set_trace()
        xhash = x.astype(np.int).tostring()
        return self.masks[xhash]


if __name__ == "__main__":
    # Test guided dropconnect
    test = SpecificGDCEncoding(sizeinputonehot=5, nrow=5, ncol=10)
    nu = np.zeros((1, 5), dtype=np.int)
    test.guided_dropconnect(nu)
    one = copy.deepcopy(nu)
    one[0, 1] = 1
    test.guided_dropconnect(one)
    two = copy.deepcopy(one)
    two[0, 2] = 1
    test.guided_dropconnect(two)


    # Test guided dropout
    test = SpecificGDOEncoding(sizeinputonehot=5, sizeout=20)
    nu = np.zeros((1, 5), dtype=np.int)
    test.guided_dropconnect(nu)
    one = copy.deepcopy(nu)
    one[0, 1] = 1
    test.guided_dropconnect(one)
    two = copy.deepcopy(one)
    two[0, 2] = 1
    test.guided_dropconnect(two)