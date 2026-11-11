package com.cjcrafter.tensor4j.data;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.TensorBuilder;
import com.google.common.base.Preconditions;

/**
 * A fixed-capacity, circular {@link Dataset} for reinforcement-learning transitions (e.g.
 * {@code (board, actionTaken, reward)} for self-play): {@link #add} appends one sample per
 * field, overwriting the oldest sample once {@link #capacity()} is reached.
 *
 * <p>Wrap the result in a {@link DataLoader} (with a {@link RandomSampler}) each training
 * step to draw shuffled minibatches from whatever has been collected so far.
 */
public final class ReplayBuffer implements Dataset {

    private final Tensor[] fields;
    private final int capacity;
    private int cursor;
    private int count;

    /**
     * Creates a replay buffer with one field per entry in {@code sampleShapes}, each
     * describing the shape of a single sample for that field (excluding the leading
     * capacity dimension), e.g. {@code new int[]{12, 8, 8}} for a board plane or
     * {@code new int[]{1}} for a scalar reward.
     * @param capacity the maximum number of samples to hold
     * @param sampleShapes the shape of one sample for each field (excluding the leading capacity dimension)
     */
    public ReplayBuffer(int capacity, int[]... sampleShapes) {
        Preconditions.checkArgument(capacity > 0, "capacity must be positive, got %s", capacity);
        Preconditions.checkArgument(sampleShapes.length > 0, "at least one field shape required");

        this.capacity = capacity;
        this.fields = new Tensor[sampleShapes.length];
        for (int i = 0; i < sampleShapes.length; i++) {
            int[] dims = sampleShapes[i];
            Preconditions.checkArgument(dims.length > 0, "field %s: sample shape must not be empty", i);

            int[] fullDims = new int[dims.length + 1];
            fullDims[0] = capacity;
            System.arraycopy(dims, 0, fullDims, 1, dims.length);
            fields[i] = TensorBuilder.builder().shape(fullDims).zeros();
        }
    }

    /**
     * Appends one transition, overwriting the oldest entry once the buffer is at capacity.
     *
     * @param sample One tensor per field, in the same order as the shapes given to the
     *               constructor. Each must have the same total element count as one row of
     *               its field (its exact shape doesn't matter, e.g. a scalar can be a
     *               {@code (1, 1)} tensor).
     */
    public void add(Tensor... sample) {
        Preconditions.checkArgument(sample.length == fields.length,
                "expected %s fields, got %s", fields.length, sample.length);

        for (int i = 0; i < fields.length; i++) {
            Tensor field = fields[i];
            int rowSize = field.getShape().numel() / capacity;
            Preconditions.checkArgument(sample[i].getShape().numel() == rowSize,
                    "field %s: expected %s elements per sample, got %s",
                    i, rowSize, sample[i].getShape().numel());

            Tensor value = sample[i].isContiguous() ? sample[i] : sample[i].contiguous();
            System.arraycopy(value.getStorage().array(), value.getOffset(),
                    field.getStorage().array(), cursor * rowSize, rowSize);
        }

        cursor = (cursor + 1) % capacity;
        count = Math.min(count + 1, capacity);
    }

    /**
     * Returns the maximum number of samples this buffer holds before it starts overwriting
     * the oldest entries.
     *
     * @return the computed value
     */
    public int capacity() {
        return capacity;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor[] tensors() {
        if (count == capacity)
            return fields;

        int[] indices = new int[count];
        for (int i = 0; i < count; i++) indices[i] = i;

        Tensor[] view = new Tensor[fields.length];
        for (int i = 0; i < fields.length; i++)
            view[i] = fields[i].indexSelect(0, indices);
        return view;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int size() {
        return count;
    }
}
