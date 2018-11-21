import sys

from TensorflowHelpers.Experiment import ExpData, Exp, ExpSaverParam
from TensorflowHelpers.Graph import ExpGraphOneXOneY, ExpGraph, ComplexGraph
from TensorflowHelpers.DataHandler import ExpCSVDataReader, ExpTFrecordsDataReader, ExpNpyDataReader
from TensorflowHelpers.ANN import ResidualBlock, DenseBlock
from GuidedDropout.ANNGuidedDrop import ComplexGraphWithGD
import numpy as np
import tensorflow as tf

class SpecificOneVarEncoding:
    def __init__(self):
        pass

    def __call__(self, x):
        res = tf.py_func(func=self.onevar_npy, inp=[x], Tout=tf.float32, name="onevar-encoding")
        res.set_shape((x.get_shape()[0], 1))
        return res

    def onevar_npy(self, x):
        """
        :param x: a numpy two dimensional array 
        :return: 
        """
        # x = x[0]
        res = np.apply_along_axis(self.onevar_npy_tmp, 1, x)
        res = res.astype(np.float32)
        return res.reshape(x.shape[0], 1)

    def onevar_npy_tmp(self, x):
        """
        :param x: a 1-D array to transform in "one var"
        :return: 
        """
        tmp = np.where(x == 1.)[0]
        if tmp.shape[0] == 0:
            tmp = -1
        return 1. + tmp

# my_exp.sess.run([my_exp.graph.input[:,0], my_exp.graph.data['deco_enco']])
import pdb
if __name__ == "__main__":
    #TODO set the seeds, generate random data and industrialize this test

    testGDO = True
    testGDC = False
    if testGDO or testGDC:
        # from TensorflowHelpers.ANNGuidedDrop import ComplexGraphWithGD

        pathdata = "/home/bdonnot/Documents/PyHades2/tfrecords_118_5000"
        path_exp = "/home/bdonnot/Documents/PyHades2/Test"
        sizes = {"prod_p": 54, "prod_q": 54, "prod_v": 54,
                 "loads_p": 99, "loads_q": 99, "loads_v": 99,
                 "deco_enco": 186, "flows_MW": 186, "flows_a": 186}

        # pathdata = "/home/benjamin/Documents/PyHades2/tfrecords_30_10000"
        # path_exp = "/home/benjamin/Documents/PyHades2/Test"
        # sizes = {"prod_p": 6, "prod_q": 6, "prod_v": 6,
        #          "loads_p": 20, "loads_q": 20, "loads_v": 20,
        #          "deco_enco": 41, "flows_MW": 41, "flows_a": 41}
        # define the experiment parameters

        parameters = ExpSaverParam(name_exp="testGDO" if testGDO else "testGDC",
                                   path=path_exp,
                                   pathdata=pathdata,
                                   num_epoch=1,
                                   num_savings=1,
                                   num_savings_minibatch=5,
                                   num_savings_model=1,
                                   continue_if_exists=True,
                                   batch_size=50)
        var_x_name = {"prod_p", "prod_v", "loads_p", "loads_q"} #, "deco_enco"}
        var_y_name = {"prod_q", "loads_v", "flows_MW", "flows_a"}
        kwargsTdata = {"filename": "N1-train.tfrecord",
                       "num_thread": 2}
        kwargsVdata = {"filename": "N1-val_small.tfrecord",
                       "num_thread": 2}
        datakwargs = {"classData": ExpTFrecordsDataReader,
                      "kwargsTdata": kwargsTdata,
                      "kwargsVdata": kwargsVdata,
                      "sizes": sizes,
                      "donnotcenter": {"deco_enco"}
                      }

        size_net = 300+3*sizes["flows_a"]
        proba_select = 3*sizes["flows_a"]/size_net
        if testGDO:
            tf.reset_default_graph()
            my_exp = Exp(parameters=parameters,
                         dataClass=ExpData, datakwargs=datakwargs,
                         graphType=ComplexGraphWithGD,
                         graphkwargs={"kwargsNN": {"layersizes": [size_net, size_net], "weightnorm": False},
                                      "var_x_name": var_x_name,
                                      "var_y_name": var_y_name,
                                      "dropout_spec": {"proba_select": proba_select, "vars": ["deco_enco"]},
                                      "outputsize": size_net,
                                      "sizes": sizes
                                      },
                         otherdsinfo={},
                         startfromscratch=True
                         )

        if testGDC:
            tf.reset_default_graph()
            my_exp = Exp(parameters=parameters,
                         dataClass=ExpData, datakwargs=datakwargs,
                         graphType=ComplexGraphWithGD,
                         graphkwargs={"kwargsNN": {"layersizes": [150, 150], "weightnorm": False},
                                      "var_x_name": var_x_name,
                                      "var_y_name": var_y_name,
                                      "dropconnect_spec": {"proba_select": 0.25, "vars": ["deco_enco"], "nrow": 150},
                                      "outputsize": 150,
                                      "sizes": sizes
                                      },
                         otherdsinfo={},
                         startfromscratch=True
                         )
        my_exp.start()
    else:
        raise RuntimeError("You should run at least one test!")

