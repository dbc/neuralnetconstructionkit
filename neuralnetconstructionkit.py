"""Very simple-minded neural net construction kit
for Tensorflow -- becuase I got tired of all the
tedius book keeping in about one afternoon.  I
was a total Tensorflow beginner the day I started
this module, so don't be surprised if it is 
re-inventing several wheels and only marginally
useful.  I did it mainly as a learning exercise.""" 

# Copyright (c) 2017 David B. Curtis
# MIT license -- see LICENSE file.
#

from functools import reduce
from math import ceil 
import tensorflow as tf
from tensorflow.contrib.layers import flatten

# To Do:
#  complete the Concat stage
#    -- test with multiscale
#    -- test with deep/stacked convolutions
#  Sigmoid stage
#  Slice stage
#  DepthwiseConv2d
#  Add average to pooling

def product_reduction(vect):
    return reduce(lambda x,y: x*y, vect)


class Stage:
    """Base class for stages of a Convolutional Neural Network.
    Mostly abstract, with some utility methods."""
    # Flat stages have an "output_width" property.
    # Convolution stages have an "output_shape" property.

    VALID_PADDING=frozenset(['SAME','VALID'])
    VALID_POOLING=frozenset(['MAX'])
    VALID_DIMS=[1,3]

    def __init__(self, name):
        self.name = str(name)

    @property
    def num_parameters(self):
        return 0
    
    @property
    def output_shape(self):
        return self._output_shape

    @output_shape.setter
    def output_shape(self, shape):
        try:
            self._output_shape = [int(x) for x in shape]
        except TypeError:
            self._output_shape = [int(shape)]
        if len(self._output_shape) not in self.VALID_DIMS:
            raise ValueError('Invalid number of dimensions.')

    def _validate_init_control(self, v):
        """0, None, or (mu,sigma).  None implies default mu,sigma."""
        if v is None or v == 0:
            return v
        if len(v) == 2:
            return tuple([float(x) for x in v])

    def _tuple_from_one_or_two_ints(self, v):
        """Return a tuple of two ints from 1 or 2 int-ish things."""
        try:
            a, b = [int(x) for x in v]
        except TypeError:
            a, b = int(v), int(v)
        return (a,b)

    @property
    def padding(self):
        return self._padding

    @padding.setter
    def padding(self, v):
        if v not in self.VALID_PADDING:
            raise ValueError (''.join(
                ['Padding must be one of:' ','.join([x for x in 
                    self.VALID_PADDING])]))
        self._padding = v

    @property
    def pooling(self):
        return self._pooling 

    @pooling.setter
    def pooling(self, v):
        if v not in self.VALID_POOLING:
            raise ValueError (''.join(
                ['Pooling must be one of:' ','.join([x for x in 
                    self.VALID_POOLING])]))
        self._pooling = v 

    # The constructors for Tensorflow variables and operators
    # memoize the results so that each one is only constructed
    # once.  Actual construction is delecated to the child
    # classes via the properties:
    # _make_weights
    # _make_biases
    # _make_network
    # The method _make_param_var is proviced as a convenience.
    @property
    def weights(self):
        try:
            return self._weights
        except AttributeError:
            self._weights = self._make_weights()
        return self._weights

    @property
    def biases(self):
        try:
            return self._biases
        except AttributeError:
            self._biases = self._make_biases()
        return self._biases

    def _make_param_var(self, shape, init, name=None):
        if init is None:
            v = tf.Variable(tf.random_normal(shape), name=name)
        elif init == 0:
            v = tf.Variable(tf.zeros(shape), name=name)
        else:
            mu, sigma = init
            v = tf.Variable(tf.random_normal(shape, mean=mu, stddev=sigma),
                name=name)
        return v

    @property
    def placeholders(self):
        try:
            return self._placeholders
        except AttributeError:
            self._placeholders = self._make_placeholders()
        return self._placeholders

    def _make_placeholders(self):
        return []

    def network(self, feed):
        try:
            return self._network
        except AttributeError:
            self._network = self._make_network(feed)
        try:
            return self._network
        except AttributeError:
            print ('Failed to create network for', self.name)
            raise

    def __repr__(self):
        arg_str = [repr(x) for x in self.repr_args()]
        arg_str.extend(['='.join([name, repr(val)]) for name, val in 
            self.repr_kwargs() if val is not None])
        args = ','.join(arg_str)
        return ''.join([self.__class__.__name__, '(', args, ')' ])

    def repr_args(self):
        return []

    def repr_kwargs(self):
        return []


class InputStage(Stage):
    """Input stage for neural network."""
    def __init__(self, name, shape):
        super().__init__(name)
        self.output_shape = shape

    def repr_args(self):
        return [self.name, self.output_shape]

    def _make_network(self, feed):
        return feed

    def feed_placeholder(self):
        shape = [None]
        shape.extend(self.output_shape)
        return [(self.name,
            tf.placeholder(tf.float32, tuple(shape), name=self.name)
            )]

        
        
class FullyConnectedLayer(Stage):
    """Specification of a fully connected layer."""
    VALID_DIMS=[1] # Fully connected layers must be flat.

    def __init__(self, name, output_width, input_layer,
            w_init=None, b_init=None):
        super().__init__(name)
        self.output_shape = output_width
        self.input_layer = input_layer
        if len(self.input_layer.output_shape) != 1:
            raise ValueError(' '.join(['Input to fully connected layer', 
                self.name, 'is not flat.']))
        self.input_width = self.input_layer.output_shape[0]
        self.w_init = self._validate_init_control(w_init)
        self.b_init = self._validate_init_control(b_init)

    def repr_args(self):
        return [self.name, self.output_shape, self.input_layer.name]

    def repr_kwargs(self):
        return [('w_init', self.w_init), ('b_init', self.b_init)]

    def _make_weights(self):
        # Get shape of weight matrix.
        shape = [self.input_width, self.output_shape[0]]
        # Build weight matrix.
        return self._make_param_var(shape, self.w_init,
            name='_'.join([self.name, 'weights']))

    def _make_biases(self):
        return self._make_param_var([self.output_shape[0]], self.b_init,
            name='_'.join([self.name, 'biases']))

    @property    
    def num_parameters(self):
        return (self.input_width + 1) * self.output_shape[0]

    def _make_network(self, feed):
        return tf.add(tf.matmul(self.input_layer.network(feed),
            self.weights, name='_'.join([self.name, 'mul'])), 
            self.biases, name='_'.join([self.name, 'add']))

    
class Flatten(Stage):
    def __init__(self, name, input_layer):
        super().__init__(name)
        self.input_layer = input_layer
        in_shape = self.input_layer.output_shape
        if len(in_shape) == 1:
            raise ValueError(''.join([input_layer.name, ' is not flattenable.']))
        else:
            self.output_shape = [product_reduction(in_shape)]

    def repr_args(self):
        return [self.name, self.input_layer.name]

    def _make_network(self, feed):
        return flatten(self.input_layer.network(feed))


class Concat(Stage):
    def __init__(self, name, input_layers):
        super().__init__(name)
        self.input_layers = input_layers 
        #self.output_shape =  # FIXME: implement

    def _make_network(self, feed):
        return tf.concat([lyr.network(feed) 
            for lyr in self.input_layers], 1, name=self.name)

    def repr_args(self):
        l = [self.name]
        l.extend([lyr.name for lyr in self.input_layers])
        return l


class Dropout(Stage):
    def __init__(self, name, input_layer):
        super().__init__(name)
        self.input_layer = input_layer
        self.output_shape = self.input_layer.output_shape

    def _make_placeholders(self):
        # probability to keep units
        ph_name = '_'.join([self.name, 'keep_prob'])
        self.ph_op = tf.placeholder(tf.float32, name=ph_name) 
        return [(ph_name, self.ph_op)]

    def _make_network(self, feed):
        feed_name, keep_prob = self.placeholders[0]
        return tf.nn.dropout(self.input_layer.network(feed), 
            keep_prob, name='_'.join([self.name, 'dropout']))

class PoolingStage(Stage):
    def __init__(self, name, strategy, input_layer, k, padding):
        super().__init__(name)
        self.pooling = strategy
        self.input_layer = input_layer
        self.k = int(k)
        self.padding = padding
        self.output_shape = self._compute_out_shape()

    def _compute_out_shape(self):
        in_width, in_height, in_depth = self.input_layer.output_shape
        if self.padding == 'VALID':
            out_width = in_width // self.k
            out_height = in_height // self.k
            return [out_width, out_height, in_depth]
        else:
            raise ValueError('Only VALID padding is implemented.')

    def _make_network(self, feed):
        if self.pooling == 'MAX':
            return tf.nn.max_pool(self.input_layer.network(feed),
                ksize=[1, self.k, self.k, 1], 
                strides=[1, self.k, self.k, 1],
                padding=self.padding,
                name='_'.join([self.name,'pool']))
        else:
            raise ValueError('Only MAX pooling is implemented.')


class ConvolutionLayer(Stage):
    VALID_DIMS=[3]
    def __init__(self, name, input_layer, depth, 
        aperture, strides, padding, w_init=None, b_init=None):
        super().__init__(name)
        self.input_layer = input_layer
        self.input_shape = self.input_layer.output_shape
        self.depth = int(depth)
        self.aperture = self._tuple_from_one_or_two_ints(aperture)
        self.strides = self._tuple_from_one_or_two_ints(strides)
        self.padding = padding
        self.w_init = self._validate_init_control(w_init)
        self.b_init = self._validate_init_control(b_init)
        self.output_shape = self._compute_out_shape()

    def _compute_out_shape(self):
        # From "Convolutional Nerual nets: 7. Quiz: Feature Map Sizes"
        #
        # SAME padding equation:
        # out_height = ceil(float(in_height) / float(strides[1]))
        # out_width  = ceil(float(in_width) / float(strides[2]))
        #
        # VALID padding equation:
        # out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
        # out_width  = ceil(float(in_width - filter_width + 1) / float(strides[2]))
        in_width, in_height = self.input_shape[0:2]
        x_stride, y_stride = self.strides
        if self.padding == 'SAME':
            width = ceil(float(in_width) / float(x_stride))
            height = ceil(float(in_height) / float(y_stride))
        elif self.padding == 'VALID':
            filter_width, filter_height = self.aperture
            width  = ceil(float(in_width - filter_width + 1) / float(x_stride))
            height = ceil(float(in_height - filter_height + 1) / float(y_stride))
        else:
            raise ValueError('Invalid padding spec.')
        return [width, height, self.depth]

    def _make_weights(self):
        # Get width and height of aperture.
        shape = list(self.aperture[:])
        # Append the output depth of the previous layer.
        shape.append(self.input_shape[2]) 
        # Append the output depth of this layer.
        shape.append(self.depth)
        return self._make_param_var(shape, self.w_init,
            name='_'.join([self.name, 'weights']))

    def _make_biases(self):
        return self._make_param_var([self.output_shape[2]], self.b_init,
            name='_'.join([self.name, 'biases']))

    @property
    def num_parameters(self):
        # Weights count depends on filter aperture and input depth, 
        # and add 1 for bias.
        l = list(self.aperture[:])
        l.append(self.input_shape[2])
        weights = product_reduction(l) + 1
        # tf.nn.conv2d output volume depends on output depth.
        return weights * self.depth

    def _make_network(self, feed):
        w_stride, h_stride = self.strides
        node = tf.nn.conv2d(self.input_layer.network(feed), 
            self.weights, strides=[1, w_stride, h_stride, 1], 
            padding=self.padding,
            name='_'.join([self.name, 'conv2d']))
        return tf.nn.bias_add(node, self.biases)


class ReluLayer(Stage):
    def __init__(self, name, input_layer):
        super().__init__(name)
        self.input_layer = input_layer
        self.output_shape = input_layer.output_shape

    def _make_network(self, feed):
        return tf.nn.relu(self.input_layer.network(feed),
            name='_'.join([self.name, 'relu']))


class NeuralNetSpec:
    def __init__(self, name, input_shape):
        self.stages = []
        self._stage_xref = {}
        self._new_stage(InputStage('input', input_shape))
        self.name = str(name)

    def _new_stage(self, stage):
        if stage.name not in self._stage_xref:
            self.stages.append(stage)
            self._stage_xref[stage.name] = stage
        else:
            raise ValueError(''.join(["Stage name '", stage.name,
                "' already in used."]))

    def _get_stage(self, name):
        return (self._stage_xref[name]
            if name is not None else self.stages[-1])

    @property
    def num_parameters(self):
        return sum([stage.num_parameters for stage in self.stages])

    @property
    def last_layer(self):
        return self.stages[-1]

    @property
    def output_shape(self):
        return self.last_layer.output_shape

    def add_fully_connected_stage(self, name, output_width,
        input_layer=None, w_init=None, b_init=None):
        in_layer = self._get_stage(input_layer)
        # in_layers = ([self.stages[-1]] if input_layers is None else
        #     input_layers)
        self._new_stage(FullyConnectedLayer(
            name, output_width, in_layer, w_init=w_init, b_init=b_init))

    def add_convolution_stage(self, name, depth, aperture, strides, 
        input_layer=None, padding=None, w_init=None, b_init=None):
        in_layer = self._get_stage(input_layer)
        padding = 'VALID' if padding is None else padding
        self._new_stage(ConvolutionLayer(name, in_layer, depth, 
            aperture, strides, padding))

    def flatten(self, name, input_layer=None):
        in_layer = self._get_stage(input_layer)
        self._new_stage(Flatten(name, in_layer))

    def dropout(self, name, input_layer=None):
        in_layer = self._get_stage(input_layer)
        self._new_stage(Dropout(name, in_layer))

    def concat(self, name, *layers):
        try:
            input_layers = [self._stage_xref[name] for name in layers]
        except KeyError as e:
            raise ValueError(' '.join(['No layer named', repr(e.args[0])]))
        self._new_stage(Concat(name, input_layers))

    def add_relu_stage(self, name, input_layer=None):
        in_layer = self._get_stage(input_layer)
        self._new_stage(ReluLayer(name, in_layer))

    def add_pool_stage(self, name, strategy=None, input_layer=None,
        k=2, padding=None):
        in_layer = self._get_stage(input_layer)
        strategy = 'MAX' if strategy is None else strategy
        padding = 'VALID' if padding is None else padding 
        self._new_stage(PoolingStage(name, strategy, in_layer,
            k, padding))

    def network(self, feed):
        return self.last_layer.network(feed)

    @property
    def placeholders(self):
        """Returns dictionary of placeholders that must be fed.
        Only valid after network() has been run."""
        rv = dict()
        for stage in self.stages:
            for name, ph_var in stage.placeholders:
                rv[name] = ph_var
        return rv

    @property
    def input_placeholder(self):
        name, ph = self.stages[0].feed_placeholder()[0]
        return ph

    def report(self):
        fmt = "{0:12s}  {1:16s}  {2:s}"
        yield fmt.format("Layer","Shape", "Parameters")
        yield fmt.format("=====","=====", "==========")
        for stage in self.stages:
            t = stage.num_parameters
            yield fmt.format( stage.name, repr(stage.output_shape), 
                '' if t == 0 else str(t))
        yield '='.join(['Total parameters', str(self.num_parameters)])

        
if __name__ == '__main__':
    nns = NeuralNetSpec('nns', 10)
    #nns.stages.append(InputStage('input', 10))
    nns.stages.append(FullyConnectedLayer('fc1', 10, nns.stages[0]))
    print (nns.num_parameters)

    trs = NeuralNetSpec('trs', input_shape=[32,32,3])
    trs.flatten('fl1')
    trs.add_fully_connected_stage('fc1', 200, b_init=0)
    print ('Stages:', repr(trs.stages))
    print ('num params:', trs.num_parameters)

    print ('================')
    t2 = NeuralNetSpec('t2', input_shape=[32,32,3])
    t2.add_convolution_stage('c1', depth=20, aperture=8, 
        strides=2, b_init=0)
    print ('num params:', t2.num_parameters)
    t2.add_relu_stage('r1')
    print ('num params after relu:', t2.num_parameters)

    print ('++++++++++++++')
    t2.dropout('c1_do')
    #x = tf.placeholder(tf.float32, (None, 32, 32, 3), name='foo')
    x = t2.input_placeholder
    print (repr(x))
    print (t2.stages[0])
    n = t2.network(x)
    print (t2.placeholders)
    t2.concat('foo', 'input', 'c1')
