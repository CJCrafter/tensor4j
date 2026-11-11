package com.cjcrafter.tensor4j.nn.losses;

import com.cjcrafter.tensor4j.Tensor;

/**
 * A criterion combining log-softmax and negative log-likelihood, matching PyTorch's
 * {@code nn.CrossEntropyLoss}.
 *
 * <p>{@code target} may be given either way PyTorch accepts it:
 * <ul>
 *     <li>Class indices: same shape as {@code input} but with {@code dim} collapsed to
 *     size 1 (the same shape {@link Tensor#argmax(int)} returns), holding integer class
 *     indices in {@code [0, C)} as floats.</li>
 *     <li>Class probabilities: exactly the same shape as {@code input}, e.g. a one-hot or
 *     soft label distribution.</li>
 * </ul>
 *
 * @param reduction How to reduce the per-example loss.
 * @param dim The class dimension to apply softmax over; negative values count from the
 *            last dimension (e.g. {@code -1} is the last dimension).
 */
public record CrossEntropyLoss(Reduction reduction, int dim) implements Loss {

    /**
     * Creates a cross-entropy loss over the last dimension with {@link Loss.Reduction#MEAN} reduction.
     */
    public CrossEntropyLoss() {
        this(Reduction.MEAN, -1);
    }

    /**
     * Creates a cross-entropy loss over the last dimension.
     *
     * @param reduction how to reduce the per-element loss
     */
    public CrossEntropyLoss(Reduction reduction) {
        this(reduction, -1);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor forward(Tensor input, Tensor target) {
        int resolvedDim = (dim >= 0) ? dim : input.getShape().dimensions() + dim;
        Tensor logProbs = input.logSoftmax(resolvedDim);

        Tensor perEntry = target.getShape().equals(input.getShape())
                ? target.mul(logProbs).sum(resolvedDim).mul(-1f)
                : logProbs.gather(resolvedDim, target).mul(-1f);

        return reduction.reduce(perEntry);
    }
}
