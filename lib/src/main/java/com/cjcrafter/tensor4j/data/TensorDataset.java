package com.cjcrafter.tensor4j.data;

import com.cjcrafter.tensor4j.Tensor;

/**
 * A <i>small</i> fixed-size dataset made up of tensors preloaded into memory.
 */
public class TensorDataset implements Dataset {

    private final Tensor[] tensors;
    private final int size;

    /**
     * Constructs a new tensor dataset.
     *
     * @param tensors All tensors. Usually at least 2 tensors (one for
     *                training data, one for labels).
     */
    public TensorDataset(Tensor... tensors) {
        if (tensors.length == 0)
            throw new IllegalArgumentException("at least one tensor required");

        this.size = tensors[0].getShape().dim(0);
        for (Tensor t : tensors) {
            if (t.getShape().dim(0) != size)
                throw new IllegalArgumentException("all tensors must have the same size in dimension 0");
        }

        // Prevent people from modifying the array, just as a sanity check
        this.tensors = tensors.clone();
    }

    @Override
    public Tensor[] tensors() {
        return tensors;
    }

    @Override
    public int size() {
        return size;
    }
}
