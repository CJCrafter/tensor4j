package com.cjcrafter.tensor4j.tests.nn;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.Tensor4j;
import com.cjcrafter.tensor4j.TensorBuilder;
import com.cjcrafter.tensor4j.data.DataLoader;
import com.cjcrafter.tensor4j.data.ReplayBuffer;
import com.cjcrafter.tensor4j.data.SequentialSampler;
import com.cjcrafter.tensor4j.nn.layers.Linear;
import com.cjcrafter.tensor4j.nn.losses.HuberLoss;
import com.cjcrafter.tensor4j.nn.optimizers.Adam;
import com.cjcrafter.tensor4j.nn.optimizers.Optimizer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * End-to-end smoke test for the RL primitives added for self-play-style training: a replay
 * buffer of (state, actionTaken, reward) transitions, gather to select Q(s,a) for the action
 * actually taken, a Huber loss against a TD target, and a full backward + optimizer step.
 */
class DqnStyleUpdateSmokeTest {

    @Test
    void gatherHuberLossAndOptimizerStepComposeCorrectly() {
        Tensor4j.manualSeed(99);

        int stateDim = 4, numActions = 3;
        Linear qNetwork = new Linear(stateDim, numActions);
        Optimizer optimizer = new Adam(qNetwork.parameters(), 1e-2f);

        ReplayBuffer buffer = new ReplayBuffer(10, new int[]{stateDim}, new int[]{1}, new int[]{1});
        for (int i = 0; i < 6; i++) {
            Tensor state = Tensor4j.builder().shape(1, stateDim).randn();
            Tensor action = TensorBuilder.builder().singleton((float) (i % numActions));
            Tensor reward = TensorBuilder.builder().singleton(1f);
            buffer.add(state, action, reward);
        }
        assertEquals(6, buffer.size());

        DataLoader loader = new DataLoader(buffer, 6, new SequentialSampler(buffer.size()));
        Tensor[] batch = loader.iterator().next();
        Tensor states = batch[0];
        Tensor actionsTaken = batch[1];
        Tensor rewards = batch[2];

        float lossBefore = trainingStep(qNetwork, optimizer, states, actionsTaken, rewards);
        float lossAfter = trainingStep(qNetwork, optimizer, states, actionsTaken, rewards);

        assertTrue(lossAfter < lossBefore,
                "loss should decrease after an optimizer step on the same batch: "
                        + lossBefore + " -> " + lossAfter);
    }

    private static float trainingStep(Linear qNetwork, Optimizer optimizer,
                                      Tensor states, Tensor actionsTaken, Tensor targetReward) {
        optimizer.zeroGrad();

        Tensor qValues = qNetwork.forward(states);           // (batch, numActions)
        Tensor qTaken = qValues.gather(1, actionsTaken);      // (batch, 1)
        Tensor loss = new HuberLoss().forward(qTaken, targetReward);

        loss.backward();
        try (var ignored = Tensor4j.noGrad()) {
            optimizer.step();
        }
        return loss.item();
    }
}
