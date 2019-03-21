import numpy as np

def fullyconnect_feedforward(in_, weight, bias):
    '''
    The feedward process of fullyconnect
      input parameters:
          in_     : the intputs, shape: [number of images, number of inputs]
          weight  : the weight matrix, shape: [number of inputs, number of outputs]
          bias    : the bias, shape: [number of outputs, 1]

      output parameters:
          out     : the output of this layer, shape: [number of images, number of outputs]
    '''
    # TODO
    # begin answer
    number_image, number_inputs = in_.shape
    in_ = np.hstack((np.ones((number_image, 1)), in_))
    weight = np.concatenate((bias.T, weight), axis=0)
    out = np.matmul(in_, weight)
    # end answer

    return out

