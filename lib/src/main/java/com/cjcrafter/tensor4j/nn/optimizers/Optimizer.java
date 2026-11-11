package com.cjcrafter.tensor4j.nn.optimizers;

import com.cjcrafter.tensor4j.Tensor;
import com.google.common.base.Preconditions;

import java.util.Collection;

/**
 * Base interface for all optimizers.
 *
 * <p><b>API note:</b> parameters need to be specified as a collection with a deterministic order.
 */
public interface Optimizer {

    /**
     * Does 1 optimization step on the parameters.
     */
    void step();

    /**
     * Resets the gradient of the parameters.
     */
    void zeroGrad();

    /**
     * Changes the learning rate used by subsequent {@link #step()} calls.
     */
    void setLearningRate(float lr);

    /**
     * Rescales the gradients of {@code params} in place so their combined L2 norm is at most
     * {@code maxNorm}. Call after {@code backward()} and before {@link #step()}.
     *
     * @return The combined gradient norm before clipping.
     */
    static float clipGradNorm(Collection<Tensor> params, float maxNorm) {
        Preconditions.checkArgument(maxNorm > 0f, "maxNorm must be positive, got %s", maxNorm);

        double sumSquares = 0.0;
        for (Tensor p : params) {
            if (!p.hasGrad()) continue;
            for (float v : p.getGrad().toFloatArray())
                sumSquares += (double) v * v;
        }
        float norm = (float) Math.sqrt(sumSquares);
        if (norm <= maxNorm) return norm;

        float scale = maxNorm / (norm + 1e-6f);
        for (Tensor p : params) {
            if (!p.hasGrad()) continue;
            Tensor g = p.getGrad();
            Preconditions.checkState(g.isContiguous() && g.getOffset() == 0,
                    "gradient clipping expects contiguous gradients");
            float[] arr = g.getStorage().array();
            for (int i = 0; i < g.getShape().numel(); i++)
                arr[i] *= scale;
        }
        return norm;
    }
}
