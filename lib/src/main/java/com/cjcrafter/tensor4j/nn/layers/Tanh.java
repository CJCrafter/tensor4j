package com.cjcrafter.tensor4j.nn.layers;

import com.cjcrafter.tensor4j.Tensor;

/**
 * Applies the elementwise hyperbolic tangent activation.
 */
public class Tanh extends Module {
    @Override
    public Tensor forward(Tensor input) {
        return input.tanh();
    }
}
