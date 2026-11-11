package com.cjcrafter.tensor4j.nn.losses;

import com.cjcrafter.tensor4j.Tensor;

/**
 * A criterion that's quadratic (L2) for small errors and linear (L1) for large ones,
 * transitioning at {@code delta}. Also known as Smooth L1 loss; commonly used for TD-error
 * regression in DQN-style reinforcement learning since it's less sensitive to outliers than
 * {@link MSELoss}.
 *
 * @param reduction How to reduce the loss vector.
 * @param delta The absolute-error threshold where the loss switches from quadratic to linear.
 */
public record HuberLoss(Reduction reduction, float delta) implements Loss {

    /**
     * Creates a Huber loss with {@link Loss.Reduction#MEAN} reduction and {@code delta = 1}.
     */
    public HuberLoss() {
        this(Reduction.MEAN, 1f);
    }

    /**
     * Creates a Huber loss with {@code delta = 1}.
     * @param reduction how to reduce the per-element loss
     */
    public HuberLoss(Reduction reduction) {
        this(reduction, 1f);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor forward(Tensor input, Tensor target) {
        Tensor diff = input.sub(target);
        Tensor absDiff = diff.abs();

        Tensor quadratic = diff.square().mul(0.5f);
        Tensor linear = absDiff.sub(0.5f * delta).mul(delta);

        Tensor isSmall = absDiff.le(delta);
        Tensor loss = isSmall.mul(quadratic).add(isSmall.rsub(1f).mul(linear));
        return reduction.reduce(loss);
    }
}
