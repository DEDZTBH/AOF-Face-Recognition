from tensorflow.python.tools import inspect_checkpoint as chkp
from os import path

chkp.print_tensors_in_checkpoint_file(path.join('data', 'model', 'svm', 'tf_model', 'model.ckpt'),
                                      tensor_name='',
                                      all_tensors=True)
