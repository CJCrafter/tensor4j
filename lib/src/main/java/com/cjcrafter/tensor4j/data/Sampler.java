package com.cjcrafter.tensor4j.data;

/**
 * An interface defining how a {@link DataLoader} should select a subset of
 * samples from the {@link Dataset}.
 */
public interface Sampler {

    /**
     * Returns an array of indices corresponding to the "row" in the
     * {@link Dataset}. Returns {@code null} when there are no more batches
     * left in this epoch.
     *
     * <p>For example, a return value of {@code {0, 4, 7}} means that the
     * caller should load the first, fifth, and eighth samples from the
     * {@link Dataset}.
     *
     * @param batchSize How many samples to grab (typically a power of 2).
     * @return The indices of the samples from the {@link Dataset}.
     */
    int[] nextBatch(int batchSize);

    /**
     * After completing 1 full epoch, {@link DataLoader} will call this method
     * to handle resetting the sampler. For example, to construct a new shuffle
     * and to set the current batch number to 0.
     */
    default void reset() {
        // intentionally empty by default
    }
}
