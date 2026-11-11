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
     *
     * @param device the device to use
     * @return the backend
     */
    public static Backend forDevice(Device device) {
        Preconditions.checkNotNull(device, "device");
        Preconditions.checkArgument(device.isCpu(), "no backend registered for %s", device);
        return CpuBackend.instance();
    }
}
