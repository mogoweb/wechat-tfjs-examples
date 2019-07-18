
import * as tf from '@tensorflow/tfjs-layers';
import * as tfc from '@tensorflow/tfjs-core';

// Tiny TFJS train / predict example.
const build_model = epochs => {
  // Create a simple model.
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

  // Prepare the model for training: Specify the loss and the optimizer.
  model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

  // Generate some synthetic data for training. (y = 2x - 1)
  const xs = tfc.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
  const ys = tfc.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1]);

  // Train the model using the data.
  model.fit(xs, ys, { epochs: epochs }).then(() => {
    model.predict(tfc.tensor2d([20], [1, 1])).print();
  });
}

module.exports = {
  build_model: build_model
}