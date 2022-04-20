# modnet_inference
  使用c++ onnxruntime实现的modnet。~~由于本人技术原因，输入尺寸必须要resize到长宽相同，不然mask输出就有问题~~
  然而由于cpu效率太低700ms/frame， 我也不想修复上述bug。
