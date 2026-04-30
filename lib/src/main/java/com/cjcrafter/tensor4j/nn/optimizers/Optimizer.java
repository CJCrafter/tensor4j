package com.cjcrafter.tensor4j.nn.optimizers;

/**
 * Base interface for all optimizers.
 *
 * @apiNote
 * Parameters need to be specified as a collection with a deterministic order.
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
}
