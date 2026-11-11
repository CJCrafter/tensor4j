package com.cjcrafter.tensor4j.data;

import com.cjcrafter.tensor4j.Tensor;

import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * Iterates a {@link Dataset} in batches, using a {@link Sampler} to pick each batch's indices.
 * Each iteration yields one {@code Tensor[]} per {@link Dataset#tensors()} entry, sliced to
 * the current batch. Resets the sampler (reshuffling, if applicable) at the start of each
 * new {@link #iterator()}.
 *
 * @param dataset   The dataset to iterate.
 * @param batchSize The number of samples per batch.
 * @param sampler   Determines the order (and, for random samplers, the shuffle) of samples.
 * @param dropLast  If {@code true}, a final undersized batch is dropped instead of yielded.
 */
public record DataLoader(
        Dataset dataset,
        int batchSize,
        Sampler sampler,
        boolean dropLast
) implements Iterable<Tensor[]> {

    /**
     * Creates a loader that keeps the final undersized batch of each epoch.
     *
     * @param dataset the dataset to draw samples from
     * @param batchSize the number of samples per batch
     * @param sampler the sampler that chooses each batch's indices
     */
    public DataLoader(Dataset dataset, int batchSize, Sampler sampler) {
        this(dataset, batchSize, sampler, false);
    }

    /**
     * Resets the sampler and returns an iterator over one epoch's worth of batches.
     */
    @Override
    public Iterator<Tensor[]> iterator() {
        sampler.reset();
        return new Iterator<>() {
            private Tensor[] next = advance();

            @Override
            public boolean hasNext() {
                return next != null;
            }

            @Override
            public Tensor[] next() {
                if (next == null) throw new NoSuchElementException();
                Tensor[] current = next;
                next = advance();
                return current;
            }

            private Tensor[] advance() {
                int[] indices = sampler.nextBatch(batchSize);
                if (indices == null) return null;
                if (dropLast && indices.length < batchSize) return null;

                Tensor[] tensors = dataset.tensors();
                Tensor[] batch = new Tensor[tensors.length];
                for (int i = 0; i < tensors.length; i++) {
                    batch[i] = tensors[i].indexSelect(0, indices);
                }
                return batch;
            }
        };
    }
}
