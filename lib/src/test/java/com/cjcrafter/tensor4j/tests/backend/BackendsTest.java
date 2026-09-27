package com.cjcrafter.tensor4j.tests.backend;

import com.cjcrafter.tensor4j.Device;
import com.cjcrafter.tensor4j.backend.Backend;
import com.cjcrafter.tensor4j.backend.Backends;
import com.cjcrafter.tensor4j.backend.cpu.CpuBackend;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class BackendsTest {

    @Test
    void cpuDeviceResolvesToCpuBackendSingleton() {
        Backend backend = Backends.forDevice(Device.cpu());
        assertSame(CpuBackend.instance(), backend);
        assertEquals(Device.cpu(), backend.device());
        assertNotNull(backend.f32Kernels());
        assertNotNull(backend.allocator());
    }

    @Test
    void nullDeviceRejected() {
        assertThrows(NullPointerException.class, () -> Backends.forDevice(null));
    }
}
