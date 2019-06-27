/**
 * @license
 * Copyright 2019 mogoweb@gmail.com. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http:// www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================
 */

// This tiny example illustrates how little code is necessary save a model in TensorFlow.js.

import * as tf from '../../tfjs/tf.min.js'

const save_model = async model => {

  const saveResult = await model.save('mp://' + wx.env.USER_DATA_PATH + '/mymodel');
  console.log(saveResult);
}

const load_model = async () => {

  let model_files = [wx.env.USER_DATA_PATH + '/mymodel.json', wx.env.USER_DATA_PATH + '/mymodel.weights.bin'];
  var model = await tf.loadLayersModel(tf.io.mpFiles(model_files));
  return model;
}

Page({
  data: {
    logs: []
  },
  onLoad: function () {
    console.log("getting-started onLoad")
    // Create a simple model.
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

    // Prepare the model for training: Specify the loss and the optimizer.
    model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

    // Generate some synthetic data for training. (y = 2x - 1)
    const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
    const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1]);

    // Train the model using the data.
    model.fit(xs, ys, { epochs: 20 }).then(async () => {
      console.log('Prediction from original model:');
      model.predict(tf.tensor2d([20], [1, 1])).print();
      await save_model(model);

      var loaded_model = load_model();

      console.log('Prediction from loaded model:');
      loaded_model.then(the_model => {
        the_model.predict(tf.tensor2d([20], [1, 1])).print();
      });
    });
  }
})