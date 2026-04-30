package com.cjcrafter.tensor4j.data;

import com.cjcrafter.tensor4j.Tensor;

import java.util.Iterator;
import java.util.NoSuchElementException;

public record DataLoader(
        Dataset dataset,
        int batchSize,
        Sampler sampler,
        boolean dropLast
) implements Iterable<Tensor[]> {

    public DataLoader(Dataset dataset, int batchSize, Sampler sampler) {
        this(dataset, batchSize, sampler, false);
    }

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
