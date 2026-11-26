import numpy as np
from numpy import ndarray

from dls.dnn import Dropout, Layer, Linear, Operation, ParamOperation


class ReLU(Operation):
    """Hyperbolic tangent activation function"""

    def do_output(self) -> ndarray:
        return np.clip(self.inputs, 0, None)

    def do_input_grad(self, output_grad: ndarray) -> ndarray:
        mask = self.inputs > 0
        return output_grad * mask


class Flatten(Operation):
    def do_output(self) -> ndarray:
        return self.inputs.reshape(self.inputs.shape[0], -1)

    def do_input_grad(self, output_grad: ndarray) -> ndarray:
        return output_grad.reshape(self.inputs.shape)


class Conv2DOperation(ParamOperation):
    def __init__(self, params: ndarray):
        super().__init__(params)
        self.param_size = params.shape[2]
        self.param_pad = self.param_size // 2

    def do_output(self) -> ndarray:
        """
        conv_in: [batch_size, channels, img_width, img_height]
        param: [in_channels, out_channels, fil_width, fil_height]
        """
        batch_size = self.inputs.shape[0]
        img_height = self.inputs.shape[2]
        img_size = self.inputs.shape[2] * self.inputs.shape[3]
        patch_size = self.params.shape[0] * self.params.shape[2] * self.params.shape[3]
        patches = self._get_image_patches(self.inputs)
        patches_reshaped = patches.transpose(1, 0, 2, 3, 4).reshape(
            batch_size, img_size, -1
        )
        param_reshaped = self.params.transpose(0, 2, 3, 1).reshape(patch_size, -1)
        output_reshaped = (
            np.matmul(patches_reshaped, param_reshaped)
            .reshape(batch_size, img_height, img_height, -1)
            .transpose(0, 3, 1, 2)
        )
        return output_reshaped

    def do_input_grad(self, output_grad: ndarray) -> ndarray:
        batch_size = self.inputs.shape[0]
        img_size = self.inputs.shape[2] * self.inputs.shape[3]
        img_height = self.inputs.shape[2]
        output_patches = (
            self._get_image_patches(output_grad)
            .transpose(1, 0, 2, 3, 4)
            .reshape(batch_size * img_size, -1)
        )
        param_reshaped = self.params.reshape(self.params.shape[0], -1).transpose(1, 0)
        return (
            np.matmul(output_patches, param_reshaped)
            .reshape(batch_size, img_height, img_height, self.params.shape[0])
            .transpose(0, 3, 1, 2)
        )

    def do_params_grad(self, output_grad: ndarray) -> ndarray:
        batch_size = self.inputs.shape[0]
        img_size = self.inputs.shape[2] * self.inputs.shape[3]
        in_channels = self.params.shape[0]
        out_channels = self.params.shape[1]
        in_patches_reshape = (
            self._get_image_patches(self.inputs)
            .transpose(1, 0, 2, 3, 4)
            .reshape(batch_size * img_size, -1)
            .transpose(1, 0)
        )
        out_grad_reshape = output_grad.transpose(0, 2, 3, 1).reshape(
            batch_size * img_size, -1
        )
        return (
            np.matmul(in_patches_reshape, out_grad_reshape)
            .reshape(in_channels, self.param_size, self.param_size, out_channels)
            .transpose(0, 3, 1, 2)
        )

    def _pad_1d(self, inp: ndarray) -> ndarray:
        z = np.array([0])
        z = np.repeat(z, self.param_pad)
        return np.concatenate([z, inp, z])

    def _pad_1d_batch(self, inp: ndarray) -> ndarray:
        outs = [self._pad_1d(obs) for obs in inp]
        return np.stack(outs)

    def _pad_2d_obs(self, inp: ndarray) -> ndarray:
        """Input is a 2 dimensional, square, 2D Tensor"""
        inp_pad = self._pad_1d_batch(inp)
        other = np.zeros((self.param_pad, inp.shape[0] + self.param_pad * 2))
        return np.concatenate([other, inp_pad, other])

    def _pad_2d_channel(self, inp: ndarray) -> ndarray:
        """inp has dimension [num_channels, image_width, image_height]"""
        return np.stack([self._pad_2d_obs(channel) for channel in inp])

    def _get_image_patches(self, inp: ndarray):
        imgs_batch_pad = np.stack([self._pad_2d_channel(obs) for obs in inp])
        patches = []
        img_height = imgs_batch_pad.shape[2]
        for h in range(img_height - self.param_size + 1):
            for w in range(img_height - self.param_size + 1):
                patch_ = imgs_batch_pad[
                    :, :, h : h + self.param_size, w : w + self.param_size
                ]
                patches.append(patch_)
        return np.stack(patches)


class Conv2D(Layer):
    def __init__(
        self,
        out_channels: int,
        param_size: int,
        dropout: float = 1.0,
        weight_init: str = "normal",
        activation: Operation | None = None,
        flatten: bool = False,
    ):
        super().__init__(out_channels)
        self.param_size = param_size
        self.dropout = dropout
        self.weight_init = weight_init
        self.activation = activation or Linear()
        self.flatten = flatten

    def do_setup_layer(self, inputs: ndarray):
        self.params = []
        in_channels = inputs.shape[1]
        if self.weight_init == "glorot":
            scale = 2.0 / (in_channels + self.neurons)
        else:
            scale = 1.0
        conv_param = np.random.normal(
            loc=0.0,
            scale=scale,
            size=(in_channels, self.neurons, self.param_size, self.param_size),
        )
        self.params.append(conv_param)

        self.operations = []
        self.operations.append(Conv2DOperation(conv_param))
        self.operations.append(self.activation)
        if self.flatten:
            self.operations.append(Flatten())
        if self.dropout < 1.0:
            self.operations.append(Dropout(keep_prob=self.dropout))
