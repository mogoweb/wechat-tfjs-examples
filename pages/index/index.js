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
  goMobilenet: function () {
    wx.navigateTo({
      url: '../mobilenet/mobilenet'
    })
  }
})
