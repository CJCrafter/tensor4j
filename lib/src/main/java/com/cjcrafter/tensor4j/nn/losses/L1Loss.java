package com.cjcrafter.tensor4j.nn.losses;

import com.cjcrafter.tensor4j.Tensor;

/**
 * A criterion that measures the absolute error between each element in the
 * {@code input} and the {@code target}.
 *
 * @param reduction How to reduce the loss vector.
 */
public record L1Loss(Reduction reduction) implements Loss {

    public L1Loss() {
        this(Reduction.MEAN);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor forward(Tensor input, Tensor target) {
        Tensor diff = input.sub(target).abs();
        return reduction.reduce(diff);
    }
}
