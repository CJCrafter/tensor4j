package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Shape;
import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.TensorBuilder;
import com.cjcrafter.tensor4j.backend.Backends;
import com.cjcrafter.tensor4j.backend.F32KernelTable;
import com.cjcrafter.tensor4j.memory.Storage;
import com.cjcrafter.tensor4j.ops.Ops;
import com.google.common.base.Preconditions;

/**
 * Backward for 2D cross-correlation ("convolution" in ML frameworks):
 * {@code out = conv2d(input, weight) + bias}, NCHW layout.
 */
public class Conv2dFn extends TensorFunction {

    private final int stride;
    private final int padding;

    /**
     * Convolves {@code input} (shape {@code (N, Cin, H, W)}) with {@code weight} (shape
     * {@code (Cout, Cin, KH, KW)}) plus {@code bias} (shape {@code (1, Cout)}), and records
     * the backward function.
     */
    public static Tensor apply(Tensor input, Tensor weight, Tensor bias, int stride, int padding) {
        int[] inDims = input.getShape().dims();
        int[] wDims = weight.getShape().dims();
        Preconditions.checkArgument(inDims.length == 4 && wDims.length == 4,
                "conv2d expects 4D (N,C,H,W) input and (Cout,Cin,KH,KW) weight, got %sD and %sD",
                inDims.length, wDims.length);
        Preconditions.checkArgument(inDims[1] == wDims[1],
                "conv2d: input channels %s don't match weight in-channels %s", inDims[1], wDims[1]);

        int outH = (inDims[2] + 2 * padding - wDims[2]) / stride + 1;
        int outW = (inDims[3] + 2 * padding - wDims[3]) / stride + 1;
        Preconditions.checkArgument(outH > 0 && outW > 0,
                "conv2d: kernel/stride/padding produce a non-positive output size");
        Shape resultShape = new Shape(inDims[0], wDims[0], outH, outW);

        Storage out = Ops.conv2d(input, weight, bias, stride, padding, resultShape);
        return Tensor.wrapResult(out, resultShape, new Conv2dFn(input, weight, bias, stride, padding));
    }

    /**
     * Creates the backward node for this operation; prefer {@code apply} over calling this directly.
     */
    public Conv2dFn(Tensor input, Tensor weight, Tensor bias, int stride, int padding) {
        super(input, weight, bias);
        this.stride = stride;
        this.padding = padding;
    }

    /**
     * Computes the gradient of each input from {@code gradOutput}.
     */
    @Override
    public Tensor[] backward(Tensor gradOutput) {
        Tensor input = inputs[0];
        Tensor weight = inputs[1];
        Tensor bias = inputs[2];

        if (!gradOutput.isContiguous()) gradOutput = gradOutput.contiguous();
        Tensor in = input.isContiguous() ? input : input.contiguous();
        Tensor w = weight.isContiguous() ? weight : weight.contiguous();

        int[] inDims = in.getShape().dims();
        int[] wDims = w.getShape().dims();
        int[] outDims = gradOutput.getShape().dims();

        Tensor gradInput = TensorBuilder.builder().shape(inDims).zeros();
        Tensor gradWeight = TensorBuilder.builder().shape(wDims).zeros();
        Tensor gradBias = TensorBuilder.builder().shape(bias.getShape()).zeros();

        F32KernelTable k = Backends.forDevice(input.device()).f32Kernels();
        k.conv2dBackward(
                in.getStorage(), in.getOffset(), inDims[0], inDims[1], inDims[2], inDims[3],
                w.getStorage(), w.getOffset(), wDims[0], wDims[2], wDims[3],
                gradOutput.getStorage(), gradOutput.getOffset(), outDims[2], outDims[3],
                gradInput.getStorage(), gradInput.getOffset(),
                gradWeight.getStorage(), gradWeight.getOffset(),
                gradBias.getStorage(), gradBias.getOffset(),
                stride, padding);

        return new Tensor[]{gradInput, gradWeight, gradBias};
    }
}
