package com.cjcrafter.tensor4j.tests;

import com.cjcrafter.tensor4j.DType;
import com.cjcrafter.tensor4j.backend.cpu.CpuF32Kernels;
import com.cjcrafter.tensor4j.memory.HeapStorage;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class MaxPool2dKernelTest {

    private static HeapStorage store(float... data) {
        HeapStorage s = HeapStorage.allocate(DType.F32, data.length);
        System.arraycopy(data, 0, s.array(), 0, data.length);
        return s;
    }

    @Test
    void poolsNonOverlapping2x2Windows() {
        // 1x1x4x4
        HeapStorage input = store(
                1, 3, 2, 4,
                5, 6, 8, 7,
                9, 12, 11, 10,
                13, 14, 16, 15);
        HeapStorage out = HeapStorage.allocate(DType.F32, 4);
        HeapStorage argmax = HeapStorage.allocate(DType.F32, 4);

        CpuF32Kernels.instance().maxPool2d(
                input, 0, 1, 1, 4, 4,
                out, 0, argmax, 0,
                2, 2, 2, 2);

        assertArrayEquals(new float[]{6, 8, 14, 16}, out.array());
        assertArrayEquals(new float[]{3, 2, 3, 2}, argmax.array());
    }

    @Test
    void backwardScattersGradientToArgmaxPosition() {
        HeapStorage argmax = store(3, 2, 3, 2);
        HeapStorage gradOutput = store(100, 200, 300, 400);
        HeapStorage gradInput = HeapStorage.allocate(DType.F32, 16); // zero-initialized 4x4

        CpuF32Kernels.instance().maxPool2dBackward(
                gradOutput, 0, argmax, 0,
                1, 1, 2, 2,
                gradInput, 0, 4, 4,
                2, 2);

        float[] expected = new float[16];
        expected[5] = 100;  // window (0,0), windowIdx 3 -> (kh=1,kw=1) -> flat index 1*4+1=5
        expected[6] = 200;  // window (0,1), windowIdx 2 -> (kh=1,kw=0) -> ih=1,iw=2 -> flat 1*4+2=6
        expected[13] = 300; // window (1,0), windowIdx 3 -> ih=3,iw=1 -> flat 3*4+1=13
        expected[14] = 400; // window (1,1), windowIdx 2 -> ih=3,iw=2 -> flat 3*4+2=14
        assertArrayEquals(expected, gradInput.array());
    }
}
