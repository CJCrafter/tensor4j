package com.cjcrafter.tensor4j.nn.layers;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.autograd.MaxPool2dFn;

/**
 * 2D max pooling over {@code (N, C, H, W)} input: each {@code kernelSize x kernelSize}
 * window is reduced to its maximum.
 */
public class MaxPool2d extends Module {

    private final int kernelSize;
    private final int stride;

    /**
     * Creates a max-pooling layer with {@code stride = kernelSize} (non-overlapping windows).
     * @param kernelSize the side length of the square kernel
     */
    public MaxPool2d(int kernelSize) {
        this(kernelSize, kernelSize);
    }

    /**
     * Creates a max-pooling layer.
     *
     * @param kernelSize the side length of the square kernel
     * @param stride the step between window positions
     */
    public MaxPool2d(int kernelSize, int stride) {
        this.kernelSize = kernelSize;
        this.stride = stride;
    }

    @Override
    public Tensor forward(Tensor input) {
        return MaxPool2dFn.apply(input, kernelSize, stride);
    }
}
