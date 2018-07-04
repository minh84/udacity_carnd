# **Semantic Segmentation**
The goals / steps of this project is to
* Implement a classifier to label the pixels of a road image using the Fully Convolutional Network (FCN) described in the [Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) by Jonathan Long, Even Shelhamer, and Trevor Darrel.
* The dataset is used in this project is the [Kiti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php).

## Rubric Points
Here we will consider the [rubric points](https://review.udacity.com/#!/rubrics/989/view) individually and describe how we addressed each point in my implementation.

## Implementation
Most of the code is already provided in `helper.py`, we only need to implement the following functions in `main.py`

*  `load_vgg`: is used to load pre-trained `VGG16` model. This can be done via [`tf.saved_model.loader.load`](https://www.tensorflow.org/api_docs/python/tf/saved_model/loader/load), then we can extract specific layers via `graph.get_tensor_by_name` see [Graph](https://www.tensorflow.org/api_docs/python/tf/Graph) api for more detail.

* `layers`: is used to create **skip-connection** layers as described in the lecture. Here we use [`conv 1x1`](https://www.tensorflow.org/api_docs/python/tf/layers/conv2d) (to preserve spatial information) to  and [`conv2d_transpose`](https://www.tensorflow.org/api_docs/python/tf/layers/conv2d_transpose) to upsample a layer to a higher resolution or dimension.

* `optimize`: is used to create loss function (`tf.nn.softmax_cross_entropy_with_logits`) and training operator (`tf.train.AdamOptimizer`).

* `train_nn`: is where the training implemented.

* `run`: is used to create FCN layers, run the training and apply trained models on test images.

The implementation is straightforward, however to make the model works one needs to
* use regularizer on kernel weights (to prevent overfitting)
* use normal initializer with small stddev.

## Training results
The loss in training is visualized bellow
