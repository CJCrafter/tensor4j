package com.cjcrafter.tensor4j.memory;

import com.cjcrafter.tensor4j.DType;

/**
 * Device-agnostic storage allocator.
 */
public interface Allocator {

    /**
     * Allocates a {@link Storage} of {@code numElements} elements of {@code dtype}.
     */
    Storage allocate(DType dtype, int numElements);
}
