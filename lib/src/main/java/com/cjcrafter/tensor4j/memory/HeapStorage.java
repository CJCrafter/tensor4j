package com.cjcrafter.tensor4j.memory;

import com.cjcrafter.tensor4j.Device;

/**
 * Storage backed by a Java heap array. Always lives on the CPU device.
 */
public abstract class HeapStorage implements Storage {

    @Override
    public final Device device() {
        return Device.cpu();
    }
}
