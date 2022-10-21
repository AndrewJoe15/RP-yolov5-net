# Yolov5Net-Faster
YOLOv5 object detection with ML.NET, ONNX

![example](https://github.com/mentalstack/yolov5-net/blob/master/img/result.jpg?raw=true)

## Runtime dependencies

For CPU usage run this line from Package Manager Console:

```
Install-Package Microsoft.ML.OnnxRuntime -Version 1.10.0
```

For GPU usage run this line from Package Manager Console:

```
Install-Package Microsoft.ML.OnnxRuntime.Gpu -Version 1.10.0
```

CPU and GPU packages can't be installed together.

OpenCvSharp4 runtime.Install a proper package corresponding to your environment.
```
Install-Package OpenCvSharp4.runtime.win -Version 4.1.1.20191216
``` 

## Usage

Yolov5Net contains two COCO pre-defined models: YoloCocoP5Model, YoloCocoP6Model. 

If you have custom trained model, then inherit from YoloModel and override all the required properties and methods. See YoloCocoP5Model or YoloCocoP6Model implementation to get know how to wrap your own model. 

```c#
using var image = Image.FromFile("Assets/test.jpg");

using var scorer = new YoloScorer<YoloCocoP5Model>("Assets/Weights/yolov5s.onnx");

List<YoloPrediction> predictions = scorer.Predict(image);

using var graphics = Graphics.FromImage(image);

foreach (var prediction in predictions) // iterate predictions to draw results
{
	double score = Math.Round(prediction.Score, 2);

	graphics.DrawRectangles(new Pen(prediction.Label.Color, 1),
		new[] { prediction.Rectangle });

	var (x, y) = (prediction.Rectangle.X - 3, prediction.Rectangle.Y - 23);

	graphics.DrawString($"{prediction.Label.Name} ({score})",
		new Font("Arial", 16, GraphicsUnit.Pixel), new SolidBrush(prediction.Label.Color),
		new PointF(x, y));
}

image.Save("Assets/result.jpg");
```

## What is the difference? Where is fast?
I use OpenCl to do the floating-point operations that convert the image data to the tensor that is the input of the inference.
The time cost is reduced from about 30ms to about 8ms.

I use OpenCv to resize image.The time cost is reduced from about 11ms to about 5ms.

Reduce nested Parallels to one Parallel.The time cost is reduced from about 10ms to about 5ms.

I run in CPU mode and the average FPS is about 23.
My device info:
CPU	11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz   2.42 GHz
RAM	16.0 GB (15.7 GB 可用)
系统类型	64 位操作系统, 基于 x64 的处理器
版本	Windows 11 家庭中文版
版本	21H2
GPU: Intel Iris Xe Graphics