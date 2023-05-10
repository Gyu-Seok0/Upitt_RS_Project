
�root"_tf_keras_network*¾{"name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}, "name": "input_6", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 8, 128]}}, "name": "reshape", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_2", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [8, 4, 128]}}, "name": "reshape_2", "inbound_nodes": [[["input_5", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape_3", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [8, 4, 128]}}, "name": "reshape_3", "inbound_nodes": [[["input_6", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_5", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_4", "inbound_nodes": [[["reshape_2", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 8, 128]}}, "name": "reshape_1", "inbound_nodes": [[["input_4", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_7", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_7", "inbound_nodes": [[["input_4", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_6", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_6", "inbound_nodes": [[["reshape_3", 0, 0, {}]]]}, {"class_name": "MeanHinAggregator", "config": {"name": "mean_hin_aggregator", "trainable": true, "dtype": "float32", "output_dim": 32, "bias": true, "act": "relu", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "bias_constraint": null}, "name": "mean_hin_aggregator", "inbound_nodes": [[["dropout_1", 0, 0, {}], ["dropout", 0, 0, {}]], [["dropout_7", 0, 0, {}], ["dropout_6", 0, 0, {}]]]}, {"class_name": "MeanHinAggregator", "config": {"name": "mean_hin_aggregator_1", "trainable": true, "dtype": "float32", "output_dim": 32, "bias": true, "act": "relu", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "bias_constraint": null}, "name": "mean_hin_aggregator_1", "inbound_nodes": [[["dropout_3", 0, 0, {}], ["dropout_2", 0, 0, {}]], [["dropout_5", 0, 0, {}], ["dropout_4", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_3", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["reshape_1", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_4", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 8, 32]}}, "name": "reshape_4", "inbound_nodes": [[["mean_hin_aggregator_1", 1, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_5", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 8, 32]}}, "name": "reshape_5", "inbound_nodes": [[["mean_hin_aggregator", 1, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_9", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_9", "inbound_nodes": [[["mean_hin_aggregator", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_8", "inbound_nodes": [[["reshape_4", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_11", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_11", "inbound_nodes": [[["mean_hin_aggregator_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_10", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_10", "inbound_nodes": [[["reshape_5", 0, 0, {}]]]}, {"class_name": "MeanHinAggregator", "config": {"name": "mean_hin_aggregator_2", "trainable": true, "dtype": "float32", "output_dim": 16, "bias": true, "act": "linear", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "bias_constraint": null}, "name": "mean_hin_aggregator_2", "inbound_nodes": [[["dropout_9", 0, 0, {}], ["dropout_8", 0, 0, {}]]]}, {"class_name": "MeanHinAggregator", "config": {"name": "mean_hin_aggregator_3", "trainable": true, "dtype": "float32", "output_dim": 16, "bias": true, "act": "linear", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "bias_constraint": null}, "name": "mean_hin_aggregator_3", "inbound_nodes": [[["dropout_11", 0, 0, {}], ["dropout_10", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_6", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [16]}}, "name": "reshape_6", "inbound_nodes": [[["mean_hin_aggregator_2", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_7", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [16]}}, "name": "reshape_7", "inbound_nodes": [[["mean_hin_aggregator_3", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAQAAABTAAAAcw4AAAB0AGoBfABkA2QCjQJTACkETukBAAAAKQHaBGF4aXPp\n/////ykC2gFL2gxsMl9ub3JtYWxpemUpAdoBeKkAcgcAAAD6Xi9ob21lL2d5dXNlb2svYW5hY29u\nZGEzL2VudnMvaGluc2FnZS9saWIvcHl0aG9uMy42L3NpdGUtcGFja2FnZXMvc3RlbGxhcmdyYXBo\nL2xheWVyL2hpbnNhZ2UucHnaCDxsYW1iZGE+ZgEAAPMAAAAA\n", null, null]}, "function_type": "lambda", "module": "stellargraph.layer.hinsage", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda", "inbound_nodes": [[["reshape_6", 0, 0, {}]], [["reshape_7", 0, 0, {}]]]}, {"class_name": "LinkEmbedding", "config": {"name": "link_embedding", "trainable": true, "dtype": "float32", "activation": "linear", "method": "ip", "axis": -2}, "name": "link_embedding", "inbound_nodes": [[["lambda", 0, 0, {}], ["lambda", 1, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "name": "activation", "inbound_nodes": [[["link_embedding", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_8", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1]}}, "name": "reshape_8", "inbound_nodes": [[["activation", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0], ["input_3", 0, 0], ["input_4", 0, 0], ["input_5", 0, 0], ["input_6", 0, 0]], "output_layers": [["reshape_8", 0, 0]]}, "shared_object_id": 42, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1, 128]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1, 128]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 8, 128]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 8, 128]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 32, 128]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 32, 128]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1, 128]}, {"class_name": "TensorShape", "items": [null, 1, 128]}, {"class_name": "TensorShape", "items": [null, 8, 128]}, {"class_name": "TensorShape", "items": [null, 8, 128]}, {"class_name": "TensorShape", "items": [null, 32, 128]}, {"class_name": "TensorShape", "items": [null, 32, 128]}], "is_graph_network": true, "full_save_spec": {"class_name": "__tuple__", "items": [[[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1, 128]}, "float32", "input_1"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1, 128]}, "float32", "input_2"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 8, 128]}, "float32", "input_3"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 8, 128]}, "float32", "input_4"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 32, 128]}, "float32", "input_5"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 32, 128]}, "float32", "input_6"]}]], {}]}, "save_spec": [{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1, 128]}, "float32", "input_1"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1, 128]}, "float32", "input_2"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 8, 128]}, "float32", "input_3"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 8, 128]}, "float32", "input_4"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 32, 128]}, "float32", "input_5"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 32, 128]}, "float32", "input_6"]}], "keras_version": "2.6.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}, "name": "input_6", "inbound_nodes": [], "shared_object_id": 2}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": [], "shared_object_id": 3}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 8, 128]}}, "name": "reshape", "inbound_nodes": [[["input_3", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "Reshape", "config": {"name": "reshape_2", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [8, 4, 128]}}, "name": "reshape_2", "inbound_nodes": [[["input_5", 0, 0, {}]]], "shared_object_id": 5}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": [], "shared_object_id": 6}, {"class_name": "Reshape", "config": {"name": "reshape_3", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [8, 4, 128]}}, "name": "reshape_3", "inbound_nodes": [[["input_6", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 8}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["reshape", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_5", "inbound_nodes": [[["input_3", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_4", "inbound_nodes": [[["reshape_2", 0, 0, {}]]], "shared_object_id": 11}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": [], "shared_object_id": 12}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 8, 128]}}, "name": "reshape_1", "inbound_nodes": [[["input_4", 0, 0, {}]]], "shared_object_id": 13}, {"class_name": "Dropout", "config": {"name": "dropout_7", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_7", "inbound_nodes": [[["input_4", 0, 0, {}]]], "shared_object_id": 14}, {"class_name": "Dropout", "config": {"name": "dropout_6", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_6", "inbound_nodes": [[["reshape_3", 0, 0, {}]]], "shared_object_id": 15}, {"class_name": "MeanHinAggregator", "config": {"name": "mean_hin_aggregator", "trainable": true, "dtype": "float32", "output_dim": 32, "bias": true, "act": "relu", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "bias_regularizer": null, "bias_constraint": null}, "name": "mean_hin_aggregator", "inbound_nodes": [[["dropout_1", 0, 0, {}], ["dropout", 0, 0, {}]], [["dropout_7", 0, 0, {}], ["dropout_6", 0, 0, {}]]], "shared_object_id": 18}, {"class_name": "MeanHinAggregator", "config": {"name": "mean_hin_aggregator_1", "trainable": true, "dtype": "float32", "output_dim": 32, "bias": true, "act": "relu", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "bias_regularizer": null, "bias_constraint": null}, "name": "mean_hin_aggregator_1", "inbound_nodes": [[["dropout_3", 0, 0, {}], ["dropout_2", 0, 0, {}]], [["dropout_5", 0, 0, {}], ["dropout_4", 0, 0, {}]]], "shared_object_id": 21}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_3", "inbound_nodes": [[["input_2", 0, 0, {}]]], "shared_object_id": 22}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["reshape_1", 0, 0, {}]]], "shared_object_id": 23}, {"class_name": "Reshape", "config": {"name": "reshape_4", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 8, 32]}}, "name": "reshape_4", "inbound_nodes": [[["mean_hin_aggregator_1", 1, 0, {}]]], "shared_object_id": 24}, {"class_name": "Reshape", "config": {"name": "reshape_5", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 8, 32]}}, "name": "reshape_5", "inbound_nodes": [[["mean_hin_aggregator", 1, 0, {}]]], "shared_object_id": 25}, {"class_name": "Dropout", "config": {"name": "dropout_9", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_9", "inbound_nodes": [[["mean_hin_aggregator", 0, 0, {}]]], "shared_object_id": 26}, {"class_name": "Dropout", "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_8", "inbound_nodes": [[["reshape_4", 0, 0, {}]]], "shared_object_id": 27}, {"class_name": "Dropout", "config": {"name": "dropout_11", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_11", "inbound_nodes": [[["mean_hin_aggregator_1", 0, 0, {}]]], "shared_object_id": 28}, {"class_name": "Dropout", "config": {"name": "dropout_10", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_10", "inbound_nodes": [[["reshape_5", 0, 0, {}]]], "shared_object_id": 29}, {"class_name": "MeanHinAggregator", "config": {"name": "mean_hin_aggregator_2", "trainable": true, "dtype": "float32", "output_dim": 16, "bias": true, "act": "linear", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 30}, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 31}, "bias_regularizer": null, "bias_constraint": null}, "name": "mean_hin_aggregator_2", "inbound_nodes": [[["dropout_9", 0, 0, {}], ["dropout_8", 0, 0, {}]]], "shared_object_id": 32}, {"class_name": "MeanHinAggregator", "config": {"name": "mean_hin_aggregator_3", "trainable": true, "dtype": "float32", "output_dim": 16, "bias": true, "act": "linear", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 33}, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 34}, "bias_regularizer": null, "bias_constraint": null}, "name": "mean_hin_aggregator_3", "inbound_nodes": [[["dropout_11", 0, 0, {}], ["dropout_10", 0, 0, {}]]], "shared_object_id": 35}, {"class_name": "Reshape", "config": {"name": "reshape_6", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [16]}}, "name": "reshape_6", "inbound_nodes": [[["mean_hin_aggregator_2", 0, 0, {}]]], "shared_object_id": 36}, {"class_name": "Reshape", "config": {"name": "reshape_7", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [16]}}, "name": "reshape_7", "inbound_nodes": [[["mean_hin_aggregator_3", 0, 0, {}]]], "shared_object_id": 37}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAQAAABTAAAAcw4AAAB0AGoBfABkA2QCjQJTACkETukBAAAAKQHaBGF4aXPp\n/////ykC2gFL2gxsMl9ub3JtYWxpemUpAdoBeKkAcgcAAAD6Xi9ob21lL2d5dXNlb2svYW5hY29u\nZGEzL2VudnMvaGluc2FnZS9saWIvcHl0aG9uMy42L3NpdGUtcGFja2FnZXMvc3RlbGxhcmdyYXBo\nL2xheWVyL2hpbnNhZ2UucHnaCDxsYW1iZGE+ZgEAAPMAAAAA\n", null, null]}, "function_type": "lambda", "module": "stellargraph.layer.hinsage", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda", "inbound_nodes": [[["reshape_6", 0, 0, {}]], [["reshape_7", 0, 0, {}]]], "shared_object_id": 38}, {"class_name": "LinkEmbedding", "config": {"name": "link_embedding", "trainable": true, "dtype": "float32", "activation": "linear", "method": "ip", "axis": -2}, "name": "link_embedding", "inbound_nodes": [[["lambda", 0, 0, {}], ["lambda", 1, 0, {}]]], "shared_object_id": 39}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "name": "activation", "inbound_nodes": [[["link_embedding", 0, 0, {}]]], "shared_object_id": 40}, {"class_name": "Reshape", "config": {"name": "reshape_8", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1]}}, "name": "reshape_8", "inbound_nodes": [[["activation", 0, 0, {}]]], "shared_object_id": 41}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0], ["input_3", 0, 0], ["input_4", 0, 0], ["input_5", 0, 0], ["input_6", 0, 0]], "output_layers": [["reshape_8", 0, 0]]}}, "training_config": {"loss": "binary_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "binary_accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 49}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.004999999888241291, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}2
�root.layer-0"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "input_3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 8, 128]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}2
�root.layer-1"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "input_5", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 128]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}}2
�root.layer-2"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "input_6", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 128]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}}2
�root.layer-3"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 128]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}2
�root.layer-4"_tf_keras_layer*�{"name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 8, 128]}}, "inbound_nodes": [[["input_3", 0, 0, {}]]], "shared_object_id": 4}2
�root.layer-5"_tf_keras_layer*�{"name": "reshape_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Reshape", "config": {"name": "reshape_2", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [8, 4, 128]}}, "inbound_nodes": [[["input_5", 0, 0, {}]]], "shared_object_id": 5}2
�root.layer-6"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "input_4", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 8, 128]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}}2
�root.layer-7"_tf_keras_layer*�{"name": "reshape_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Reshape", "config": {"name": "reshape_3", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [8, 4, 128]}}, "inbound_nodes": [[["input_6", 0, 0, {}]]], "shared_object_id": 7}2
�	root.layer-8"_tf_keras_layer*�{"name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 8}2
�
root.layer-9"_tf_keras_layer*�{"name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["reshape", 0, 0, {}]]], "shared_object_id": 9}2
�
�
�
�
�
�
�root.layer_with_weights-0"_tf_keras_layer*�{"name": "mean_hin_aggregator", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MeanHinAggregator", "config": {"name": "mean_hin_aggregator", "trainable": true, "dtype": "float32", "output_dim": 32, "bias": true, "act": "relu", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "bias_regularizer": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_1", 0, 0, {}], ["dropout", 0, 0, {}]], [["dropout_7", 0, 0, {}], ["dropout_6", 0, 0, {}]]], "shared_object_id": 18, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1, 128]}, {"class_name": "TensorShape", "items": [null, 1, 8, 128]}]}2
�root.layer_with_weights-1"_tf_keras_layer*�{"name": "mean_hin_aggregator_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MeanHinAggregator", "config": {"name": "mean_hin_aggregator_1", "trainable": true, "dtype": "float32", "output_dim": 32, "bias": true, "act": "relu", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "bias_regularizer": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_3", 0, 0, {}], ["dropout_2", 0, 0, {}]], [["dropout_5", 0, 0, {}], ["dropout_4", 0, 0, {}]]], "shared_object_id": 21, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1, 128]}, {"class_name": "TensorShape", "items": [null, 1, 8, 128]}]}2
�
�
�
�
�
�
�
�
�root.layer_with_weights-2"_tf_keras_layer*�{"name": "mean_hin_aggregator_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MeanHinAggregator", "config": {"name": "mean_hin_aggregator_2", "trainable": true, "dtype": "float32", "output_dim": 16, "bias": true, "act": "linear", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 30}, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 31}, "bias_regularizer": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_9", 0, 0, {}], ["dropout_8", 0, 0, {}]]], "shared_object_id": 32, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1, 32]}, {"class_name": "TensorShape", "items": [null, 1, 8, 32]}]}2
�root.layer_with_weights-3"_tf_keras_layer*�{"name": "mean_hin_aggregator_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MeanHinAggregator", "config": {"name": "mean_hin_aggregator_3", "trainable": true, "dtype": "float32", "output_dim": 16, "bias": true, "act": "linear", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 33}, "kernel_regularizer": null, "kernel_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 34}, "bias_regularizer": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_11", 0, 0, {}], ["dropout_10", 0, 0, {}]]], "shared_object_id": 35, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1, 32]}, {"class_name": "TensorShape", "items": [null, 1, 8, 32]}]}2
�
�
�
� 
�!
�"
��root.keras_api.metrics.0"_tf_keras_metric*�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 50}2
��root.keras_api.metrics.1"_tf_keras_metric*�{"class_name": "MeanMetricWrapper", "name": "binary_accuracy", "dtype": "float32", "config": {"name": "binary_accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 49}2