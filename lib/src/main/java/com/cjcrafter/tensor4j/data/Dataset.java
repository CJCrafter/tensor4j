package com.cjcrafter.tensor4j.data;

import com.cjcrafter.tensor4j.Tensor;

/**
 * The dataset you are training on. For a typical ML task with a labeled
 * dataset (e.g. MNIST), you can use a {@link TensorDataset}:
 * <ol>
 *     <li>{@code (60000, 28, 28)} (e.g. 60,000 samples of 28x28 images)</li>
 *     <li>{@code (60000, 1)} (e.g. labels for each image)</li>
 * </ol>
 *
 * @see TensorDataset
 */
public interface Dataset {

    /**
     * Returns the full current dataset. The first dimension of each tensor is
     * guaranteed to be equal (e.g. each tensor has the same number of batches).
     *
     * <p>Typically, this is just 2 tensors of shape (batch, x) and (batch, y).
     * For a reinforcement learning task, like Q-Learning, this might be your
     * replay buffer. So you would have 5 aligned tensors for state, action,
     * next state, reward, and done.
     *
     * @return The current dataset/labels stored as aligned tensors.
     */
    Tensor[] tensors();

    /**
     * How many training samples are present in the dataset.
     *
     * <p>Note that this value <b>MAY</b> be constant, but is not guaranteed to
     * be constant. For example, the MNIST dataset will always return 60,000,
     * but a replay buffer in reinforcement learning may return some small
     * number to start, then grow over time.
     *
     * @return The total number of samples.
     */
    int size();
}
