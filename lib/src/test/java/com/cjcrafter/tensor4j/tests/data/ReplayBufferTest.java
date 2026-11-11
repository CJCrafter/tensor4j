package com.cjcrafter.tensor4j.tests.data;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.TensorBuilder;
import com.cjcrafter.tensor4j.data.ReplayBuffer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ReplayBufferTest {

    private static final float EPS = 1e-6f;

    private static Tensor scalar(float v) {
        return TensorBuilder.builder().singleton(v);
    }

    @Test
    void sizeGrowsUntilCapacityThenStaysThere() {
        ReplayBuffer buffer = new ReplayBuffer(3, new int[]{1});
        assertEquals(0, buffer.size());
        assertEquals(3, buffer.capacity());

        buffer.add(scalar(1f));
        assertEquals(1, buffer.size());
        buffer.add(scalar(2f));
        buffer.add(scalar(3f));
        assertEquals(3, buffer.size());
        buffer.add(scalar(4f));
        assertEquals(3, buffer.size(), "size should not exceed capacity");
    }

    @Test
    void overwritesOldestEntryOnceFull() {
        ReplayBuffer buffer = new ReplayBuffer(3, new int[]{1});
        buffer.add(scalar(1f));
        buffer.add(scalar(2f));
        buffer.add(scalar(3f));
        buffer.add(scalar(4f)); // overwrites the "1"

        Tensor values = buffer.tensors()[0];
        float[] flat = values.toFloatArray();
        java.util.Arrays.sort(flat);
        assertArrayEquals(new float[]{2f, 3f, 4f}, flat, EPS);
    }

    @Test
    void multipleFieldsStayAligned() {
        // (board-like feature, scalar reward) pairs
        ReplayBuffer buffer = new ReplayBuffer(2, new int[]{2}, new int[]{1});
        buffer.add(TensorBuilder.builder().shape(1, 2).fromArray(10f, 11f), scalar(100f));
        buffer.add(TensorBuilder.builder().shape(1, 2).fromArray(20f, 21f), scalar(200f));

        Tensor[] tensors = buffer.tensors();
        Tensor boards = tensors[0];
        Tensor rewards = tensors[1];

        assertEquals(10f, boards.get(0, 0), EPS);
        assertEquals(11f, boards.get(0, 1), EPS);
        assertEquals(100f, rewards.get(0, 0), EPS);

        assertEquals(20f, boards.get(1, 0), EPS);
        assertEquals(21f, boards.get(1, 1), EPS);
        assertEquals(200f, rewards.get(1, 0), EPS);
    }

    @Test
    void tensorsViewBeforeFullOnlyIncludesWrittenRows() {
        ReplayBuffer buffer = new ReplayBuffer(5, new int[]{1});
        buffer.add(scalar(7f));
        buffer.add(scalar(8f));

        Tensor values = buffer.tensors()[0];
        assertEquals(2, values.getShape().dim(0));
        assertEquals(7f, values.get(0, 0), EPS);
        assertEquals(8f, values.get(1, 0), EPS);
    }
}
