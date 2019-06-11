import * as tf from '../../tfjs/tf.min.js'

import {
  IMAGENET_CLASSES
} from './imagenet_classes';

const MOBILENET_MODEL_PATH =
  // tslint:disable-next-line:max-line-length
  'https://cnpmjs.org/mirrors/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';

const IMAGE_SIZE = 224;
const TOPK_PREDICTIONS = 10;

let that;
let mobilenet;

function downloadFile(url) {
  return new Promise((resolve, reject) => {
    wx.downloadFile({
      url: url,
      success: function (res) {
        if (res.statusCode === 200) {
          resolve(res.tempFilePath);
        } else {
          console.log('downloadFile fail, statusCode is:', res.statusCode);
          reject(false);
        }
      },
      fail: function ({
        errMsg
      }) {
        console.log('downloadFile fail, err is:', errMsg);
        reject(false);
      }
    });
  });
}

async function saveToLocal(tempFilePath, destFilePath) {
  return new Promise((resolve, reject) => {
    wx.getFileSystemManager().saveFile({
      tempFilePath: tempFilePath,
      filePath: destFilePath,
      success: (res) => {
        console.log("save to:" + res.savedFilePath);
        resolve(res.savedFilePath);
      },
      fail: (res) => {
        reject(res);
      }
    });
  });
}

const downloadWeights = async (json_file) => {
  // 解析model.json文件
  let lines = wx.getFileSystemManager().readFileSync(json_file, "utf-8");
  let json = JSON.parse(lines);
  let weights_manifest = json.weightsManifest;

  // 下载所有的权重文件，并保存到本地
  var downloadList = new Array();
  for (let i = 0; i < weights_manifest.length; i++) {
    let weight_file = MOBILENET_MODEL_PATH.substring(0, MOBILENET_MODEL_PATH.lastIndexOf("/") + 1) + weights_manifest[i].paths;
    console.log('Downloading model weight file:' + weight_file);
    downloadList.push(downloadFile(weight_file).then((tempFilePath) => {
      saveToLocal(tempFilePath, wx.env.USER_DATA_PATH + '/' + weights_manifest[i].paths);
    }));
  }

  // 期望所有权重文件都保存到本地成功
  Promise.all(downloadList).then(() => {
    console.log("download all weights success!");
    // model weights全部保存成功，将model.json文件也保存到本地
    return saveToLocal(json_file, wx.env.USER_DATA_PATH + '/model.json');
  });
}

const mobilenetDemo = async() => {
  console.log("Downloading model manifest ...");

  // 首先下载model.json
  let model_json = await downloadFile(MOBILENET_MODEL_PATH);

  // 然后根据模型定义文件下载weights
  let flag = await downloadWeights(model_json);
  /*
  console.log('Loading model...');

  mobilenet = await tf.loadLayersModel(MOBILENET_MODEL_PATH);

  // Warmup the model. This isn't necessary, but makes the first prediction
  // faster. Call `dispose` to release the WebGL memory allocated for the return
  // value of `predict`.
  mobilenet.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();

  getImageData('mobileCanvas', 'cat.jpg', function (imgData) {
    //  在此处得到的RGB数据
    console.log("getImageData");
    predict(imgData);
  });
  */
}

// 获取图像RGB数据
var getImageData = function(canvasId, imgUrl, callback, imgWidth, imgHeight) {
  console.log("entering getImageData");

  const ctx = wx.createCanvasContext(canvasId);
  ctx.drawImage(imgUrl, 0, 0, imgWidth || IMAGE_SIZE, imgHeight || IMAGE_SIZE);
  ctx.draw(false, () => {
    console.log("ctx.draw");
    // API 1.9.0 获取图像数据
    wx.canvasGetImageData({
      canvasId: canvasId,
      x: 0,
      y: 0,
      width: imgWidth || IMAGE_SIZE,
      height: imgHeight || IMAGE_SIZE,
      success(res) {
        var result = res;
        console.log("buf:" + [result.data.buffer]);

        // RGBA to RGB
        var rgbData = new Uint8Array(res.width * res.height * 3);
        let idx = 0;
        for (let i = 0; i < res.data.length; i += 4) {
          rgbData[idx] = res.data[i];
          rgbData[idx + 1] = res.data[i + 1];
          rgbData[idx + 2] = res.data[i + 2];
          idx += 3;
        }
        callback(rgbData);
      },
      fail: e => {
        console.error(e);
      },
    })
  })
};

/**
 * Given an image element, makes a prediction through mobilenet returning the
 * probabilities of the top K classes.
 */
async function predict(imgData) {
  wx.showLoading({
    title: '正在识别图像...',
  });

  // The first start time includes the time it takes to extract the image
  // from the HTML and preprocess it, in additon to the predict() call.
  const startTime1 = performance.now();
  // The second start time excludes the extraction and preprocessing and
  // includes only the predict() call.
  let startTime2;
  const logits = tf.tidy(() => {
    // tf.browser.fromPixels() returns a Tensor from an image element.
    const img = tf.tensor3d(imgData, [IMAGE_SIZE, IMAGE_SIZE, 3]).toFloat();

    const offset = tf.scalar(127.5);
    // Normalize the image from [0, 255] to [-1, 1].
    const normalized = img.sub(offset).div(offset);

    // Reshape to a single-element batch so we can pass it to predict.
    const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);

    startTime2 = performance.now();
    // Make a prediction through mobilenet.
    return mobilenet.predict(batched);
  });

  // Convert logits to probabilities and class names.
  const classes = await getTopKClasses(logits, TOPK_PREDICTIONS);
  const totalTime1 = performance.now() - startTime1;
  const totalTime2 = performance.now() - startTime2;
  console.log(`Done in ${Math.floor(totalTime1)} ms ` +
    `(not including preprocessing: ${Math.floor(totalTime2)} ms)`);

  wx.hideLoading();

  // Show the classes.
  showResults(classes);
}

function showResults(classes) {
  let probabilities = "";
  for (let i = 0; i < classes.length; i++) {
    console.log("class:" + classes[i].className + ", probability:" + classes[i].probability.toFixed(3));
    probabilities = probabilities + classes[i].className + ": " + classes[i].probability.toFixed(3) + "\n";
  }

  that.setData({
    display: "block",
    probabilities: probabilities,
  });
}

/**
 * Computes the probabilities of the topK classes given logits by computing
 * softmax to get probabilities and then sorting the probabilities.
 * @param logits Tensor representing the logits from MobileNet.
 * @param topK The number of top predictions to show.
 */
export async function getTopKClasses(logits, topK) {
  const values = await logits.data();

  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({
      value: values[i],
      index: i
    });
  }
  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });
  const topkValues = new Float32Array(topK);
  const topkIndices = new Int32Array(topK);
  for (let i = 0; i < topK; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }

  const topClassesAndProbs = [];
  for (let i = 0; i < topkIndices.length; i++) {
    topClassesAndProbs.push({
      className: IMAGENET_CLASSES[topkIndices[i]],
      probability: topkValues[i]
    })
  }
  return topClassesAndProbs;
}

Page({

  /**
   * 页面的初始数据
   */
  data: {
    display: "none"
  },

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: function(options) {
    that = this;
    wx.showLoading({
      title: '正在加载模型...',
    });
    mobilenetDemo(predict);
  },
  /**
   * 生命周期函数--监听页面初次渲染完成
   */
  onReady: function() {},
  /**
   * 生命周期函数--监听页面显示
   */
  onShow: function() {},
})