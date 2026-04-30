package com.cjcrafter.tensor4j.data;

public class SequentialSampler implements Sampler {

    private final int size;
    private int cursor;

    public SequentialSampler(int size) {
        this.size = size;
        this.cursor = 0;
    }

    @Override
    public int[] nextBatch(int batchSize) {
        if (cursor >= size) return null;

        int n = Math.min(batchSize, size - cursor);
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) {
            indices[i] = cursor + i;
        }
        cursor += n;
        return indices;
    }

    @Override
    public void reset() {
        cursor = 0;
    }
}
