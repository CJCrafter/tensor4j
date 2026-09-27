package com.cjcrafter.tensor4j;

import com.google.common.base.Preconditions;

/**
 * Identifies a specific compute device by {@link DeviceKind} plus an ordinal.
 */
public final class Device {

    private static final Device CPU = new Device(DeviceKind.CPU, 0);

    private final DeviceKind kind;
    private final int ordinal;

    private Device(DeviceKind kind, int ordinal) {
        this.kind = Preconditions.checkNotNull(kind, "kind");
        Preconditions.checkArgument(ordinal >= 0, "ordinal must be non-negative, got %s", ordinal);
        this.ordinal = ordinal;
    }

    /**
     * Returns the {@link DeviceKind} family this device belongs to.
     */
    public DeviceKind kind() {
        return kind;
    }

    /**
     * Returns the index of this device within its {@link DeviceKind} (e.g. GPU 0 vs GPU 1).
     */
    public int ordinal() {
        return ordinal;
    }

    /**
     * Returns {@code true} if this device is the CPU.
     */
    public boolean isCpu() {
        return kind == DeviceKind.CPU;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Device d)) return false;
        return ordinal == d.ordinal && kind == d.kind;
    }

    @Override
    public int hashCode() {
        return kind.hashCode() * 31 + ordinal;
    }

    @Override
    public String toString() {
        return kind == DeviceKind.CPU ? "cpu" : kind.name().toLowerCase() + ":" + ordinal;
    }

    /**
     * Returns the singleton CPU device.
     */
    public static Device cpu() {
        return CPU;
    }
}
