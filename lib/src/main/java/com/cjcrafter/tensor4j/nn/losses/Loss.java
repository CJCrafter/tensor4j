package com.cjcrafter.tensor4j.nn.losses;

import com.cjcrafter.tensor4j.Tensor;

/**
 * The abstraction over criteria.
 *
 * <p>This abstraction is purely for convenience. All the same effects can be
 * achieved by manually doing math with your tensors after feeding them through
 * your network. For example, instead of using {@link MSELoss}, you could just
 * as easily use:
 * <pre>{@code
 *      Tensor y_hat = model.forward(x);
 *      Tensor target = some_target;
 *      Tensor loss = y_hat.sub(target).square().mean();
 * }</pre>
 * Which gives you the same result as calling {@link #forward(Tensor, Tensor)}
 * on {@link MSELoss}.
 */
public interface Loss {

    /**
     * Compares the {@code input} and the {@code target} and returns some
     * tensor (typically, <i>though not strictly</i>, a {@code 1 x 1} tensor)
     * whose value is proportional to the relative error.
     *
     * @param input The models prediction that you would like to adjust.
     * @param target The target "ground truth."
     * @return The loss value.
     */
    Tensor forward(Tensor input, Tensor target);

    /**
     * Appends the given loss function to this loss function.
     *
     * <p>This is nice for when you want to combine multiple criteria into 1
     * criterion (e.g. adding in regularization factors).
     *
     * @param other The other loss function.
     * @return The new, combined loss function.
     */
    default Loss with(Loss other) {
        Loss self = this;
        return (input, target) ->
                self.forward(input, target).add(other.forward(input, target));
    }


    /**
     * The reduction strategy to use. Typically, you always use {@link #MEAN}.
     */
    enum Reduction {
        /**
         * No reduction... e.g. return a vector of absolute differences.
         */
        NONE,

        /**
         * Mean of the absolute differences.
         */
        MEAN,

        /**
         * Sum of the absolute differences.
         */
        SUM;


        /**
         * Reduces the given {@code input} vector according to the reduction
         * scheme.
         *
         * @param input The tensor to reduce.
         * @return The reduced tensor.
         */
        public Tensor reduce(Tensor input) {
            return switch (this) {
                case NONE -> input;
                case MEAN -> input.mean();
                case SUM -> input.sum();
            };
        }
    }
}
