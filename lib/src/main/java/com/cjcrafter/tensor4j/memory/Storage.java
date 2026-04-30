package com.cjcrafter.tensor4j.memory;

import com.cjcrafter.tensor4j.DType;
import com.cjcrafter.tensor4j.Device;

/**
 * A contiguous block of memory that stores the underlying
 * data of a tensor on some device.
 */
public interface Storage {

    DType dtype();

    Device device();

    long sizeBytes();

    default long capacity() {
        return sizeBytes() / dtype().sizeBytes();
    }
}
