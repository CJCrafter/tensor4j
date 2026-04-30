package com.cjcrafter.tensor4j.data;

import com.cjcrafter.tensor4j.Tensor4j;

/**
 * Uniformly and random samples all samples from a fixed-size dataset exactly 1
 * time per epoch.
 */
public class RandomSampler implements Sampler {

    private final int size;
    private int[] permutation;
    private int cursor;

    /**
     * Constructs a new random sampler.
     *
     * @param size The total number of samples in the dataset.
     */
    public RandomSampler(int size) {
        this.size = size;
        this.permutation = Tensor4j.randperm(size);
        this.cursor = 0;
    }

    @Override
    public int[] nextBatch(int batchSize) {
        if (cursor >= size) return null;

        int n = Math.min(batchSize, size - cursor);
        int[] indices = new int[n];
        System.arraycopy(permutation, cursor, indices, 0, n);
        cursor += n;
        return indices;
    }

    @Override
    public void reset() {
        permutation = Tensor4j.randperm(size);
        cursor = 0;
    }
}
