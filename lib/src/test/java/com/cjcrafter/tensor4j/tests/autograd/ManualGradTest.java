package com.cjcrafter.tensor4j.tests.autograd;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.Tensor4j;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Some manual test cases for autograd special cases.
 */
public class ManualGradTest {

    /**
     * This test is just checking to make sure that gradients to not get
     * overridden by calculations further in the past.
     *
     * @see <a href="https://youtu.be/VMj-3S1tku0?t=4948">Accumulation Explanation by Andrej Kaparthy</a>
     */
    @Test
    void testGradientsDoNotGetOverridden() {
        Tensor a = Tensor4j.builder().requiresGrad().singleton(-2f); // grad = (b)(1) + (1)(-6) = -3
        Tensor b = Tensor4j.builder().requiresGrad().singleton(3f); // grad = (a)(1) + (1)(-6) = -8

        Tensor d = a.mul(b); // grad = e = 1
        Tensor e = a.add(b); // grad = d = -6
        Tensor f = d.mul(e);

        // f = (a * b) * (a + b)
        // df/da = d/da [ aab + abb ] = 2ab + bb = -12 + 9 = -3
        // df/db = d/db [ aab + abb ] = aa + 2ab = 4 - 12 = -8

        f.backward();

        double actualA = a.getGrad().item();
        assertEquals(-3, actualA);

        double actualB = b.getGrad().item();
        assertEquals(-8, actualB);
    }

    /**
     * If multiple gradients flow into 1 tensor "b," but tensor "a" depends on
     * b, then b must be fully calculated before calculating the gradient of a.
     */
    @Test
    void testGradientsAreFullyCalculated() {
        Tensor a = Tensor4j.builder().requiresGrad().singleton(-2f);
        Tensor b = Tensor4j.builder().requiresGrad().singleton(3f);

        Tensor c = a.add(b);   // c = 1
        Tensor d = c.mul(b);   // d = 3
        Tensor e = c.add(d);   // e = 4

        // e = c + c*b = (a+b) + (a+b)*b = a + b + ab + bb
        // de/da = 1 + b = 4
        // de/db = 1 + a + 2b = 5
        //
        // Intermediate node c has grad = 1 (from e directly) + b (from d) = 4.
        // Without topological sort, a naive DFS from e might traverse
        // e -> c -> leaves BEFORE visiting d, propagating c's gradient
        // as 1 instead of 4, producing wrong gradients for a and b.

        e.backward();

        assertEquals(4, a.getGrad().item());
        assertEquals(5, b.getGrad().item());
    }
}
