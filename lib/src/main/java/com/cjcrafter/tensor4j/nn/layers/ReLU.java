package com.cjcrafter.tensor4j.nn.layers;

import com.cjcrafter.tensor4j.Tensor;

public class ReLU extends Module {

    @Override
    public Tensor forward(Tensor input) {
        return input.max(0f);
    }
}
