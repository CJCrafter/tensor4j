package com.cjcrafter.tensor4j.tests;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.TensorBuilder;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class TensorMatmulTest {

    private static final float EPS = 1e-6f;

    @Test
    void matmul2x2() {
        // [1, 2] @ [5, 6] = [1*5+2*7, 1*6+2*8] = [19, 22]
        // [3, 4]   [7, 8]   [3*5+4*7, 3*6+4*8]   [43, 50]
        Tensor a = TensorBuilder.builder()
                .shape(2, 2)
                .fromArray(1, 2, 3, 4);
        Tensor b = TensorBuilder.builder()
                .shape(2, 2)
                .fromArray(5, 6, 7, 8);
        Tensor c = a.matmul(b);

        assertEquals(2, c.getShape().dim(0));
        assertEquals(2, c.getShape().dim(1));
        assertEquals(19f, c.get(0, 0), EPS);
        assertEquals(22f, c.get(0, 1), EPS);
        assertEquals(43f, c.get(1, 0), EPS);
        assertEquals(50f, c.get(1, 1), EPS);
    }

    @Test
    void matmulNonSquare() {
        // (2,3) @ (3,2) → (2,2)
        // [1, 2, 3] @ [7,  8 ] = [1*7+2*9+3*11,  1*8+2*10+3*12] = [58,  64]
        // [4, 5, 6]   [9,  10]   [4*7+5*9+6*11,  4*8+5*10+6*12]   [139, 154]
        //              [11, 12]
        Tensor a = TensorBuilder.builder().shape(2, 3).fromArray(1, 2, 3, 4, 5, 6);
        Tensor b = TensorBuilder.builder().shape(3, 2).fromArray(7, 8, 9, 10, 11, 12);
        Tensor c = a.matmul(b);

        assertEquals(2, c.getShape().dim(0));
        assertEquals(2, c.getShape().dim(1));
        assertEquals(58f,  c.get(0, 0), EPS);
        assertEquals(64f,  c.get(0, 1), EPS);
        assertEquals(139f, c.get(1, 0), EPS);
        assertEquals(154f, c.get(1, 1), EPS);
    }
}
