//index.js
//获取应用实例
const app = getApp()

Page({
  data: {
  },
  onLoad: function () {
  },
  goGettingStarted: function() {
    wx.navigateTo({
      url: '../getting-started/index'
    })
  },
  goMnist: function () {
    wx.navigateTo({
      url: '../mnist/index'
    })
  },
  goMobilenet: function () {
    wx.navigateTo({
      url: '../mobilenet/mobilenet'
    })
  },
  goLocalStorage: function () {
    wx.navigateTo({
      url: '../mobilenet_local/index'
    })
  },
  goSaveLocal: function () {
    wx.navigateTo({
      url: '../save_local/index'
    })
  }
})
