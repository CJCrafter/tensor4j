package com.cjcrafter.tensor4j.tests;

import com.cjcrafter.tensor4j.DType;
import com.cjcrafter.tensor4j.backend.cpu.CpuF32Kernels;
import com.cjcrafter.tensor4j.memory.NativeStorage;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MatmulKernelTest {

    private static NativeStorage store(float... data) {
        NativeStorage s = NativeStorage.allocate(DType.F32, data.length);
        MemorySegment seg = s.segment();
        for (int i = 0; i < data.length; i++)
            seg.setAtIndex(ValueLayout.JAVA_FLOAT, i, data[i]);
        s.incRef();
        return s;
    }

    private static float[] toArray(MemorySegment s, int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++)
            out[i] = s.getAtIndex(ValueLayout.JAVA_FLOAT, i);
        return out;
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

        NativeStorage aStore = store(a);
        NativeStorage bStore = store(b);
        NativeStorage outStore = NativeStorage.allocate(DType.F32, 16);
        outStore.incRef();
        // zero-init c (matmul accumulates into c)
        for (int i = 0; i < 16; i++)
            outStore.segment().setAtIndex(ValueLayout.JAVA_FLOAT, i, 0f);

        CpuF32Kernels.instance().matmul(aStore, 0, 8, 3, bStore, 0, 2, outStore, 0);

        float[] actual = toArray(outStore.segment(), 16);
        for (int i = 0; i < 16; i++) {
            assertEquals(expected[i], actual[i], 1e-4f, "mismatch at i=" + i);
        }
    }
}
