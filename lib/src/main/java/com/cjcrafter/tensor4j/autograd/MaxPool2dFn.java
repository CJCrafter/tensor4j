package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.DType;
import com.cjcrafter.tensor4j.Shape;
import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.TensorBuilder;
import com.cjcrafter.tensor4j.backend.Backend;
import com.cjcrafter.tensor4j.backend.Backends;
import com.cjcrafter.tensor4j.backend.F32KernelTable;
import com.cjcrafter.tensor4j.memory.Storage;
import com.google.common.base.Preconditions;

/**
 * Backward for 2D max pooling over {@code (N, C, H, W)} input: the upstream gradient is
 * scattered back to whichever position in each window was the max.
 */
public class MaxPool2dFn extends TensorFunction {

    private final Storage argmax;
    private final int kernelSize;
    private final int stride;
    private final int h;
    private final int w;

    /**
     * Pools {@code input} over {@code kernelSize x kernelSize} windows with the given stride
     * and records the backward function.
     */
    public static Tensor apply(Tensor input, int kernelSize, int stride) {
        if (!input.isContiguous()) input = input.contiguous();
        int[] dims = input.getShape().dims();
        Preconditions.checkArgument(dims.length == 4,
                "maxPool2d expects 4D (N,C,H,W) input, got %sD", dims.length);

        int n = dims[0], c = dims[1], h = dims[2], w = dims[3];
        int outH = (h - kernelSize) / stride + 1;
        int outW = (w - kernelSize) / stride + 1;
        Preconditions.checkArgument(outH > 0 && outW > 0,
                "maxPool2d: kernel/stride produce a non-positive output size");
        Shape resultShape = new Shape(n, c, outH, outW);

        Backend backend = Backends.forDevice(input.device());
        F32KernelTable k = backend.f32Kernels();
        Storage out = backend.allocator().allocate(DType.F32, resultShape.numel());
        Storage argmax = backend.allocator().allocate(DType.F32, resultShape.numel());

        k.maxPool2d(input.getStorage(), input.getOffset(), n, c, h, w,
                out, 0, argmax, 0, outH, outW, kernelSize, stride);

        return Tensor.wrapResult(out, resultShape,
                new MaxPool2dFn(input, argmax, kernelSize, stride, h, w));
    }

    /**
     * Creates the backward node for this operation; prefer {@code apply} over calling this directly.
     */
    public MaxPool2dFn(Tensor input, Storage argmax, int kernelSize, int stride, int h, int w) {
        super(input);
        this.argmax = argmax;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.h = h;
        this.w = w;
    }

    /**
     * Computes the gradient of each input from {@code gradOutput}.
     */
    @Override
    public Tensor[] backward(Tensor gradOutput) {
        Tensor input = inputs[0];
        if (!gradOutput.isContiguous()) gradOutput = gradOutput.contiguous();

        int[] inDims = input.getShape().dims();
        int[] outDims = gradOutput.getShape().dims();

        Tensor gradInput = TensorBuilder.builder().shape(inDims).zeros();

        F32KernelTable k = Backends.forDevice(input.device()).f32Kernels();
        k.maxPool2dBackward(gradOutput.getStorage(), gradOutput.getOffset(), argmax, 0,
                inDims[0], inDims[1], outDims[2], outDims[3],
                gradInput.getStorage(), gradInput.getOffset(), h, w,
                kernelSize, stride);

        return new Tensor[]{gradInput};
    }
}
