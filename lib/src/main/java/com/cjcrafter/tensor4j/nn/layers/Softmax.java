package com.cjcrafter.tensor4j.nn.layers;

import com.cjcrafter.tensor4j.Tensor;

public class Softmax extends Module {

    private final int dim;

    public Softmax(int dim) {
        this.dim = dim;
    }

    @Override
    public Tensor forward(Tensor input) {
        return input.softmax(dim);
    }
}
