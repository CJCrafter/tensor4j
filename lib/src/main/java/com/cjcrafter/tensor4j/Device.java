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

    public static Device cpu() {
        return CPU;
    }

    public DeviceKind kind() {
        return kind;
    }

    public int ordinal() {
        return ordinal;
    }

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
}
