import tensorflow as tf

def check_list_and_convert(the_object):
    if isinstance(the_object, list):
        return the_object
    return [the_object]


class TfMap:
    """ a container for inputs, outputs, and loss in a tf graph. This object exists only
    to make well-defined the tf inputs, outputs, and losses used in the policy_opt_tf class."""

    def __init__(self, input_tensor, target_output_tensor, precision_tensor, output_op, loss_op):
        self.input_tensor = input_tensor
        self.target_output_tensor = target_output_tensor
        self.precision_tensor = precision_tensor
        self.output_op = output_op
        self.loss_op = loss_op

    @classmethod
    def init_from_lists(cls, inputs, outputs, loss):
        inputs = check_list_and_convert(inputs)
        outputs = check_list_and_convert(outputs)
        loss = check_list_and_convert(loss)
        if len(inputs) < 3:  # pad for the constructor if needed.
            inputs += [None]*(3 - len(inputs))
        return cls(inputs[0], inputs[1], inputs[2], outputs[0], loss[0])

    def get_input_tensor(self):
        return self.input_tensor

    def set_input_tensor(self, input_tensor):
        self.input_tensor = input_tensor

    def get_target_output_tensor(self):
        return self.target_output_tensor

    def set_target_output_tensor(self, target_output_tensor):
        self.target_output_tensor = target_output_tensor

    def get_precision_tensor(self):
        return self.precision_tensor

    def set_precision_tensor(self, precision_tensor):
        self.precision_tensor = precision_tensor

    def get_output_op(self):
        return self.output_op

    def set_output_op(self, output_op):
        self.output_op = output_op

    def get_loss_op(self):
        return self.loss_op

    def set_loss_op(self, loss_op):
        self.loss_op = loss_op

def init_weights(shape, name=None):
    return tf.Variable(tf.random.normal(shape, stddev=0.01), name=name)

def init_bias(shape, name=None):
    return tf.Variable(tf.zeros(shape, dtype='float'), name=name)

def batched_matrix_vector_multiply(vector, matrix):
    """ computes x^T A in mini-batches. """
    vector_batch_as_matricies = tf.expand_dims(vector, [1])
    mult_result = tf.matmul(vector_batch_as_matricies, matrix)
    squeezed_result = tf.squeeze(mult_result, [1])
    return squeezed_result

def get_input_layer():
    """produce the placeholder inputs that are used to run ops forward and backwards.
        net_input: usually an observation.
        action: mu, the ground truth actions we're trying to learn.
        precision: precision matrix used to compute loss."""
    net_input = tf.compat.v1.placeholder("float", [None, None], name='nn_input')  # (N*T) x dO
    action = tf.compat.v1.placeholder('float', [None, None], name='action')       # (N*T) x dU
    precision = tf.compat.v1.placeholder('float', [None, None, None], name='precision') # (N*T) x dU x dU
    return net_input, action, precision

def get_loss_layer(mlp_out, action, precision, batch_size):
    """The loss layer used for the MLP network is obtained through this class."""
    scale_factor = tf.constant(2*batch_size, dtype='float')
    uP = batched_matrix_vector_multiply(action - mlp_out, precision)
    uPu = tf.reduce_sum(uP*(action - mlp_out))  # this last dot product is then summed, so we just the sum all at once.
    return uPu/scale_factor

def fully_connected_tf_network(dim_input, dim_output, batch_size=25, network_config=None):
    
    dim_hidden = network_config['dim_hidden'] + [dim_output]
    n_layers = len(dim_hidden)
    
    nn_input, action, precision = get_input_layer()
    
    weights = []
    biases = []
    in_shape = dim_input
    for layer_step in range(0, n_layers):
        cur_weight = init_weights([in_shape, dim_hidden[layer_step]], name='w_' + str(layer_step))
        cur_bias = init_bias([dim_hidden[layer_step]], name='b_' + str(layer_step))
        in_shape = dim_hidden[layer_step]
        weights.append(cur_weight)
        biases.append(cur_bias)
    
    cur_top = nn_input
    for layer_step in range(0, n_layers):
        if layer_step != n_layers-1:  # final layer has no RELU
            cur_top = tf.nn.relu(tf.matmul(cur_top, weights[layer_step]) + biases[layer_step])
        else:
            cur_top = tf.nn.relu6(tf.matmul(cur_top, weights[layer_step]) + biases[layer_step])
    
    mlp_applied = cur_top
    loss_out = get_loss_layer(mlp_out=mlp_applied, action=action, precision=precision, batch_size=batch_size)

    return TfMap.init_from_lists([nn_input, action, precision], [mlp_applied], [loss_out])
