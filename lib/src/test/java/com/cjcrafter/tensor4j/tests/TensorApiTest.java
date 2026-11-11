package com.cjcrafter.tensor4j.tests;

import com.cjcrafter.tensor4j.DType;
import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.Tensor4j;
import com.cjcrafter.tensor4j.TensorBuilder;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class TensorApiTest {

    private static final float EPS = 1e-5f;

    @Test
    void powForwardAndBackward() {
        Tensor x = TensorBuilder.builder().shape(1, 3).requiresGrad().fromArray(1f, 2f, 3f);
        Tensor y = x.pow(2f);
        assertEquals(4f, y.get(0, 1), EPS);
        y.sum().backward();
        // d/dx x^2 = 2x
        assertEquals(2f, x.getGrad().get(0, 0), EPS);
        assertEquals(4f, x.getGrad().get(0, 1), EPS);
        assertEquals(6f, x.getGrad().get(0, 2), EPS);
    }

    @Test
    void meanAlongDim() {
        Tensor x = TensorBuilder.builder().shape(2, 3).fromArray(1f, 2f, 3f, 4f, 5f, 6f);
        Tensor m = x.mean(1);
        assertArrayEquals(new int[]{2, 1}, m.getShape().dims());
        assertEquals(2f, m.get(0, 0), EPS);
        assertEquals(5f, m.get(1, 0), EPS);
    }

    @Test
    void flattenKeepsBatchDim() {
        Tensor x = Tensor4j.builder().shape(2, 3, 4, 5).randn();
        Tensor f = x.flatten();
        assertArrayEquals(new int[]{2, 60}, f.getShape().dims());
        assertEquals(x.get(1, 2, 3, 4), f.get(1, 59), EPS);
    }

    @Test
    void flattenWorksOnNonContiguousTensor() {
        Tensor x = TensorBuilder.builder().shape(2, 3).fromArray(1f, 2f, 3f, 4f, 5f, 6f).transpose();
        Tensor f = x.flatten();
        assertArrayEquals(new int[]{3, 2}, f.getShape().dims());
        assertEquals(x.get(0, 1), f.get(0, 1), EPS);
    }

    @Test
    void squeezeAndUnsqueezeRoundTrip() {
        Tensor x = Tensor4j.builder().shape(2, 3, 4).randn();
        Tensor up = x.unsqueeze(1);
        assertArrayEquals(new int[]{2, 1, 3, 4}, up.getShape().dims());
        Tensor down = up.squeeze(1);
        assertArrayEquals(new int[]{2, 3, 4}, down.getShape().dims());
        assertEquals(x.get(1, 2, 3), down.get(1, 2, 3), EPS);
    }

    @Test
    void squeezeRejectsNonUnitDim() {
        Tensor x = Tensor4j.builder().shape(2, 3, 4).randn();
        assertThrows(IllegalArgumentException.class, () -> x.squeeze(1));
    }

    @Test
    void permuteReordersDimensions() {
        Tensor x = TensorBuilder.builder().shape(2, 3, 4).randn();
        Tensor p = x.permute(2, 0, 1);
        assertArrayEquals(new int[]{4, 2, 3}, p.getShape().dims());
        assertEquals(x.get(1, 2, 3), p.get(3, 1, 2), EPS);
    }

    @Test
    void permuteBackwardRoutesGradientBack() {
        Tensor x = TensorBuilder.builder().shape(2, 3).requiresGrad().fromArray(1f, 2f, 3f, 4f, 5f, 6f);
        Tensor loss = x.permute(1, 0).mul(Tensor4j.builder().shape(3, 2).fromArray(1f, 10f, 100f, 1000f, 1e4f, 1e5f)).sum();
        loss.backward();
        // element x[i][j] lands at permuted[j][i]; grad is that position's weight
        assertEquals(1f, x.getGrad().get(0, 0), EPS);
        assertEquals(100f, x.getGrad().get(0, 1), EPS);
        assertEquals(1e5f, x.getGrad().get(1, 2), EPS);
    }

    @Test
    void permuteRejectsNonPermutation() {
        Tensor x = Tensor4j.builder().shape(2, 3).randn();
        assertThrows(IllegalArgumentException.class, () -> x.permute(0, 0));
    }

    @Test
    void setReturnsSelfForChaining() {
        Tensor x = Tensor4j.builder().shape(2, 2).zeros();
        assertSame(x, x.set(5f, 0, 1));
        assertEquals(5f, x.get(0, 1), EPS);
    }

    @Test
    void hasGradTracksBackward() {
        Tensor x = TensorBuilder.builder().shape(1, 2).requiresGrad().fromArray(1f, 2f);
        assertFalse(x.hasGrad());
        x.sum().backward();
        assertTrue(x.hasGrad());
    }

    @Test
    void zerosAndOnesLikeMatchShape() {
        Tensor x = Tensor4j.builder().shape(2, 3).randn();
        Tensor z = Tensor4j.zerosLike(x);
        Tensor o = Tensor4j.onesLike(x);
        assertArrayEquals(x.getShape().dims(), z.getShape().dims());
        assertEquals(0f, z.get(1, 2), EPS);
        assertEquals(1f, o.get(1, 2), EPS);
    }

    @Test
    void onlyFloat32IsSupported() {
        assertEquals(4, DType.F32.sizeBytes());
        assertEquals("float32", DType.F32.displayName());
    }

    @Test
    void amaxAminAlongDim() {
        Tensor x = TensorBuilder.builder().shape(2, 3).fromArray(1f, 5f, 3f, 4f, 2f, 6f);
        assertEquals(5f, x.amax(1).get(0, 0), EPS);
        assertEquals(6f, x.amax(1).get(1, 0), EPS);
        assertEquals(1f, x.amin(1).get(0, 0), EPS);
        assertEquals(2f, x.amin(1).get(1, 0), EPS);
    }

    @Test
    void fullMaxAndMinReduceToScalar() {
        Tensor x = TensorBuilder.builder().shape(2, 3).fromArray(1f, 5f, -3f, 4f, 2f, 6f);
        assertEquals(6f, x.max().item(), EPS);
        assertEquals(-3f, x.min().item(), EPS);
    }

    @Test
    void fullMaxBackwardRoutesToArgmax() {
        Tensor x = TensorBuilder.builder().shape(2, 2).requiresGrad().fromArray(1f, 9f, 3f, 4f);
        x.max().backward();
        assertEquals(1f, x.getGrad().get(0, 1), EPS);
        assertEquals(0f, x.getGrad().get(0, 0), EPS);
    }

    @Test
    void argminReturnsIndexOfMinimum() {
        Tensor x = TensorBuilder.builder().shape(2, 3).fromArray(1f, 5f, 3f, 4f, 2f, 6f);
        assertEquals(0f, x.argmin(1).get(0, 0), EPS);
        assertEquals(1f, x.argmin(1).get(1, 0), EPS);
        assertEquals(1f, x.argmax(1).get(0, 0), EPS);
    }
}
