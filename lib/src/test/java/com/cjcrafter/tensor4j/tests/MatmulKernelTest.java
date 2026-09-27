package com.cjcrafter.tensor4j.tests;

import com.cjcrafter.tensor4j.DType;
import com.cjcrafter.tensor4j.backend.cpu.CpuF32Kernels;
import com.cjcrafter.tensor4j.memory.HeapStorage;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MatmulKernelTest {

    private static HeapStorage store(float... data) {
        HeapStorage s = HeapStorage.allocate(DType.F32, data.length);
        System.arraycopy(data, 0, s.array(), 0, data.length);
        return s;
    }

    @Test
    void matmul8x3times3x2() {
        // a: 8x3 (rows 0..7, each row 0..2)
        float[] a = new float[24];
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 3; j++)
                a[i * 3 + j] = i + j * 0.1f;
        // b: 3x2
        float[] b = {1f, 2f, 3f, 4f, 5f, 6f};

        // expected = a @ b
        float[] expected = new float[16];
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 2; j++) {
                float s = 0;
                for (int kk = 0; kk < 3; kk++)
                    s += a[i * 3 + kk] * b[kk * 2 + j];
                expected[i * 2 + j] = s;
            }
        }

        HeapStorage aStore = store(a);
        HeapStorage bStore = store(b);
        // fresh arrays are already zero-initialized (matmul accumulates into c)
        HeapStorage outStore = HeapStorage.allocate(DType.F32, 16);

        CpuF32Kernels.instance().matmul(aStore, 0, 8, 3, bStore, 0, 2, outStore, 0);

        float[] actual = outStore.array();
        for (int i = 0; i < 16; i++) {
            assertEquals(expected[i], actual[i], 1e-4f, "mismatch at i=" + i);
        }
    }
}
