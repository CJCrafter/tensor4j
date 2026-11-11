package com.cjcrafter.tensor4j.tests;

import com.cjcrafter.tensor4j.DType;
import com.cjcrafter.tensor4j.backend.cpu.CpuF32Kernels;
import com.cjcrafter.tensor4j.memory.HeapStorage;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class Conv2dKernelTest {

    private static HeapStorage store(float... data) {
        HeapStorage s = HeapStorage.allocate(DType.F32, data.length);
        System.arraycopy(data, 0, s.array(), 0, data.length);
        return s;
    }

    @Test
    void singleChannelValidConvolution() {
        // input: 1x1x3x3
        // 1 2 3
        // 4 5 6
        // 7 8 9
        HeapStorage input = store(1, 2, 3, 4, 5, 6, 7, 8, 9);
        // weight: 1 filter, 1 in-channel, 2x2
        // 1 0
        // 0 1
        HeapStorage weight = store(1, 0, 0, 1);
        HeapStorage bias = store(0);
        HeapStorage out = HeapStorage.allocate(DType.F32, 4); // 1x1x2x2

        CpuF32Kernels.instance().conv2d(
                input, 0, 1, 1, 3, 3,
                weight, 0, 1, 2, 2,
                bias, 0,
                out, 0, 2, 2,
                1, 0);

        // [[1+5, 2+6], [4+8, 5+9]] = [[6, 8], [12, 14]]
        assertEquals(6f, out.array()[0]);
        assertEquals(8f, out.array()[1]);
        assertEquals(12f, out.array()[2]);
        assertEquals(14f, out.array()[3]);
    }

    @Test
    void biasIsAddedToEveryOutputPosition() {
        HeapStorage input = store(1, 2, 3, 4, 5, 6, 7, 8, 9);
        HeapStorage weight = store(1, 0, 0, 1);
        HeapStorage bias = store(100);
        HeapStorage out = HeapStorage.allocate(DType.F32, 4);

        CpuF32Kernels.instance().conv2d(
                input, 0, 1, 1, 3, 3,
                weight, 0, 1, 2, 2,
                bias, 0,
                out, 0, 2, 2,
                1, 0);

        assertEquals(106f, out.array()[0]);
        assertEquals(108f, out.array()[1]);
        assertEquals(112f, out.array()[2]);
        assertEquals(114f, out.array()[3]);
    }

    @Test
    void samePaddingPreservesSpatialSize() {
        // 1x1x2x2 input, 3x3 kernel, padding=1 -> output stays 2x2 ("same" padding)
        HeapStorage input = store(1, 2, 3, 4);
        float[] identityCenter = new float[9];
        identityCenter[4] = 1f; // center tap = identity kernel
        HeapStorage weight = store(identityCenter);
        HeapStorage bias = store(0);
        HeapStorage out = HeapStorage.allocate(DType.F32, 4);

        CpuF32Kernels.instance().conv2d(
                input, 0, 1, 1, 2, 2,
                weight, 0, 1, 3, 3,
                bias, 0,
                out, 0, 2, 2,
                1, 1);

        assertEquals(1f, out.array()[0]);
        assertEquals(2f, out.array()[1]);
        assertEquals(3f, out.array()[2]);
        assertEquals(4f, out.array()[3]);
    }
}
