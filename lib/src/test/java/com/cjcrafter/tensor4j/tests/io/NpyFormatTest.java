package com.cjcrafter.tensor4j.tests.io;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.TensorBuilder;
import com.cjcrafter.tensor4j.io.NpyFormat;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

class NpyFormatTest {

    private static final float EPS = 1e-6f;

    @Test
    void roundtrip2D() throws IOException {
        float[] values = {1f, 2f, 3f, 4f, 5f, 6f};
        Tensor original = TensorBuilder.builder().shape(2, 3).fromArray(values);

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        NpyFormat.write(baos, original);

        Tensor loaded = NpyFormat.read(new ByteArrayInputStream(baos.toByteArray()));

        assertArrayEquals(new int[]{2, 3}, loaded.getShape().dims());
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                assertEquals(original.get(i, j), loaded.get(i, j), EPS);
    }

    @Test
    void roundtrip3D() throws IOException {
        Tensor original = TensorBuilder.builder().shape(2, 3, 4).fromArray(new float[24]);
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 4; k++)
                    original.set(i * 12 + j * 4 + k + 0.5f, i, j, k);

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        NpyFormat.write(baos, original);

        Tensor loaded = NpyFormat.read(new ByteArrayInputStream(baos.toByteArray()));

        assertArrayEquals(new int[]{2, 3, 4}, loaded.getShape().dims());
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 4; k++)
                    assertEquals(original.get(i, j, k), loaded.get(i, j, k), EPS);
    }

    @Test
    void roundtripPreservesNegativesAndFractions() throws IOException {
        float[] values = {-1.5f, 0f, 3.14159f, Float.MAX_VALUE};
        Tensor original = TensorBuilder.builder().shape(2, 2).fromArray(values);

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        NpyFormat.write(baos, original);

        Tensor loaded = NpyFormat.read(new ByteArrayInputStream(baos.toByteArray()));
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                assertEquals(original.get(i, j), loaded.get(i, j), EPS);
    }
}
