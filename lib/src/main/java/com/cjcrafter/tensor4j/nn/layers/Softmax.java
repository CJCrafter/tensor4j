package com.cjcrafter.tensor4j.nn.layers;

import com.cjcrafter.tensor4j.Tensor;

/**
 * Applies {@link Tensor#softmax(int)} along a fixed dimension.
 */
public class Softmax extends Module {

    private final int dim;

    /**
     * Creates a softmax layer normalizing along {@code dim}.
     */
    public Softmax(int dim) {
        this.dim = dim;
    }

    @Override
    public Tensor forward(Tensor input) {
        return input.softmax(dim);
    }
}
