package com.cjcrafter.tensor4j.backend;

import com.cjcrafter.tensor4j.Device;
import com.cjcrafter.tensor4j.backend.cpu.CpuBackend;
import com.google.common.base.Preconditions;

/**
 * Registry that resolves a {@link Backend} for a given {@link Device}.
 */
public final class Backends {

    private Backends() {}

    /**
     * Returns the backend that owns compute and allocation for the given device.
     */
    public static Backend forDevice(Device device) {
        Preconditions.checkNotNull(device, "device");
        if (device.isCpu()) {
            return CpuBackend.instance();
        }
        throw new UnsupportedOperationException("no backend registered for " + device);
    }
}
