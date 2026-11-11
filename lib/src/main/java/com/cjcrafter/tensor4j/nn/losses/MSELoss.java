package com.cjcrafter.tensor4j.nn.losses;

import com.cjcrafter.tensor4j.Tensor;

/**
 * Creates a criterion that measures the mean squared error (squared L2 norm)
 * between each element in the {@code input} and {@code target}.
 *
 * @param reduction How to reduce the loss vector.
 */
public record MSELoss(Reduction reduction) implements Loss {

    /**
     * Creates an MSE loss with {@link Loss.Reduction#MEAN} reduction.
     */
    public MSELoss() {
        this(Reduction.MEAN);
    }

    /**
     * {@inheritDoc}
     * @param input the input tensor
     * @param target the target values to compare against
     * @return the resulting tensor
     */
    public Tensor forward(Tensor input, Tensor target) {
        Tensor diff = input.sub(target).square();
        return reduction.reduce(diff);
    }
}
