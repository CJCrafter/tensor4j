package com.cjcrafter.tensor4j.tests.nn;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.Tensor4j;
import com.cjcrafter.tensor4j.nn.layers.*;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * End-to-end smoke test for the shape of chess-bot assignment this API is meant to support:
 * a shared CNN trunk over an (N, C, 8, 8) board encoding, feeding separate policy (move
 * probabilities over legal moves) and value (position eval) heads.
 */
class ChessNetSmokeTest {

    @Test
    void cnnTrunkWithPolicyAndValueHeadsFlowsShapesAndBackpropagates() {
        Tensor4j.manualSeed(42);

        int batch = 2, inChannels = 4, boardSize = 8, numMoves = 20;

        Sequential trunk = new Sequential(
                new Conv2d(inChannels, 8, 3, 1, 1), // "same" padding keeps the board at 8x8
                new BatchNorm2d(8),
                new ReLU(),
                new Conv2d(8, 8, 3, 1, 1),
                new BatchNorm2d(8),
                new ReLU()
        );
        Linear policyHead = new Linear(8 * boardSize * boardSize, numMoves);
        Linear valueHead = new Linear(8 * boardSize * boardSize, 1);

        Tensor board = Tensor4j.builder()
                .shape(batch, inChannels, boardSize, boardSize).requiresGrad().randn();

        Tensor features = trunk.forward(board);
        assertArrayEquals(new int[]{batch, 8, boardSize, boardSize}, features.getShape().dims());

        Tensor flat = features.view(batch, 8 * boardSize * boardSize);

        // Legal-move mask: only the first half of moves are legal this position.
        Tensor mask = Tensor4j.builder().shape(batch, numMoves).zeros();
        for (int b = 0; b < batch; b++)
            for (int m = 0; m < numMoves / 2; m++)
                mask.set(1f, b, m);

        Tensor policy = policyHead.forward(flat).maskedSoftmax(mask, 1);
        assertArrayEquals(new int[]{batch, numMoves}, policy.getShape().dims());
        for (int b = 0; b < batch; b++) {
            float rowSum = 0f;
            for (int m = 0; m < numMoves; m++) rowSum += policy.get(b, m);
            assertEquals(1f, rowSum, 1e-3f, "policy row should sum to 1");
            for (int m = numMoves / 2; m < numMoves; m++)
                assertEquals(0f, policy.get(b, m), 1e-4f, "illegal move should have ~0 probability");
        }

        Tensor value = valueHead.forward(flat).tanh();
        assertArrayEquals(new int[]{batch, 1}, value.getShape().dims());
        for (int b = 0; b < batch; b++) {
            float v = value.get(b, 0);
            assertTrue(v >= -1f && v <= 1f, "tanh output should be bounded in [-1, 1]");
        }

        // The whole trunk + both heads should backpropagate cleanly in one graph.
        Tensor loss = policy.sum().add(value.sum());
        assertDoesNotThrow(loss::backward);
        assertNotNull(board.getGrad());
    }
}
