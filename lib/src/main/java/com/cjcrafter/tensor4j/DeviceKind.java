package com.cjcrafter.tensor4j;

/**
 * Family of compute device. Each kind has its own memory system and kernel backend.
 */
public enum DeviceKind {

    CPU,
    CUDA,
    OPENCL;

    public boolean isCpu() {
        return this == CPU;
    }
}
