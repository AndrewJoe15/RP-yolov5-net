using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCL.Net;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Yolov5Net.Scorer.Extensions;

namespace Yolov5Net.Scorer
{
    /// <summary>
    /// Yolov5 scorer.
    /// </summary>
    public class YoloScorer : IDisposable
    {
        public YoloModel model
        {
            get => _model;
            set => _model = value;
        }

        private YoloModel _model;

        private readonly InferenceSession _inferenceSession;

        /// <summary>
        /// Outputs value between 0 and 1.
        /// </summary>
        private float Sigmoid(float value)
        {
            return 1 / (1 + (float)Math.Exp(-value));
        }

        /// <summary>
        /// Converts xywh bbox format to xyxy.
        /// </summary>
        private float[] Xywh2xyxy(float[] source)
        {
            var result = new float[4];

            result[0] = source[0] - source[2] / 2f;
            result[1] = source[1] - source[3] / 2f;
            result[2] = source[0] + source[2] / 2f;
            result[3] = source[1] + source[3] / 2f;

            return result;
        }

        /// <summary>
        /// Returns value clamped to the inclusive range of min and max.
        /// </summary>
        public float Clamp(float value, float min, float max)
        {
            return (value < min) ? min : (value > max) ? max : value;
        }

        /// <summary>
        /// Resizes image keeping ratio to fit model input size.
        /// </summary>
        private Bitmap ResizeImage(Image image)
        {
            PixelFormat format = image.PixelFormat;

            var output = new Bitmap(_model.Width, _model.Height, format);

            var (w, h) = (image.Width, image.Height); // image width and height
            var (xRatio, yRatio) = (_model.Width / (float)w, _model.Height / (float)h); // x, y ratios
            var ratio = Math.Min(xRatio, yRatio); // ratio = resized / original
            var (width, height) = ((int)(w * ratio), (int)(h * ratio)); // roi width and height
            var (x, y) = ((_model.Width / 2) - (width / 2), (_model.Height / 2) - (height / 2)); // roi x and y coordinates
            var roi = new Rectangle(x, y, width, height); // region of interest

            using (var graphics = Graphics.FromImage(output))
            {
                graphics.Clear(Color.FromArgb(0, 0, 0, 0)); // clear canvas

                graphics.SmoothingMode = SmoothingMode.None; // no smoothing
                graphics.InterpolationMode = InterpolationMode.Bilinear; // bilinear interpolation
                graphics.PixelOffsetMode = PixelOffsetMode.Half; // half pixel offset

                graphics.DrawImage(image, roi); // draw scaled
            }

            return output;
        }

        private OclCaller oclCaller;
        /// <summary>
        /// Extracts pixels into tensor for net input.
        /// </summary>
        private Tensor<float> ExtractPixels(Image image)
        {
            var bitmap = (Bitmap)image;
            int[] imgData = PrepareImgData(bitmap);
            float[] receive;
            //stopwatch.Restart();
            oclCaller.Compute(imgData, out receive);
            //Console.WriteLine("Compute:" + stopwatch.ElapsedMilliseconds);

            return new DenseTensor<float>(receive, new int[] { 1, 3, _model.Height, _model.Width });
        }

        private int[] PrepareImgData(Bitmap bitmap)
        {
            var rectangle = new Rectangle(0, 0, bitmap.Width, bitmap.Height);
            BitmapData bitmapData = bitmap.LockBits(rectangle, ImageLockMode.ReadOnly, bitmap.PixelFormat);
            try
            {
                int bytesPerPixel = Image.GetPixelFormatSize(bitmap.PixelFormat) / 8;
                int[] result = new int[1 * bytesPerPixel * _model.Height * _model.Width];

                int layer = _model.Height * _model.Width;
                unsafe // speed up conversion by direct work with memory
                {
                    Parallel.For(0, bitmapData.Height, (y) =>
                    {
                        byte* r = (byte*)bitmapData.Scan0 + (y * bitmapData.Stride);
                        for (int x = 0; x < bitmapData.Width; x++)
                        {
                            result[layer * 0 + y * _model.Width + x] = r[x * bytesPerPixel + 2];
                            result[layer * 1 + y * _model.Width + x] = r[x * bytesPerPixel + 1];
                            result[layer * 2 + y * _model.Width + x] = r[x * bytesPerPixel + 0];
                        }
                    });
                }
                return result;
            }
            finally
            {
                bitmap.UnlockBits(bitmapData);
            }
        }

        private Stopwatch stopwatch = new Stopwatch();
        /// <summary>
        /// Runs inference session.
        /// </summary>
        private DenseTensor<float>[] Inference(Image image)
        {
            Bitmap resized = null;

            if (image.Width != _model.Width || image.Height != _model.Height)
            {
                //stopwatch.Restart();
                resized = ResizeImage(image); // fit image size to specified input size
                //Console.WriteLine("Resize:" + stopwatch.ElapsedMilliseconds);
            }

            var inputs = new List<NamedOnnxValue> // add image as onnx input
            {
                NamedOnnxValue.CreateFromTensor("images", ExtractPixels(resized ?? image))
            };

            //stopwatch.Restart();
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> result = _inferenceSession.Run(inputs); // run inference
            //Console.WriteLine("Inference:" + stopwatch.ElapsedMilliseconds);

            var output = new List<DenseTensor<float>>();

            foreach (var item in _model.Outputs) // add outputs for processing
            {
                try
                {
                    // 获取结果 name == "output" name == "output0" ...
                    output.Add(result.First(x => x.Name == item).Value as DenseTensor<float>);
                }
                catch (Exception e)
                {
                    // 如果result.First() 没有找到元素 则跳过
                    if (e.GetType() == typeof(InvalidOperationException))
                    {
                        continue;
                    }
                }
            };

            return output.ToArray();
        }

        /// <summary>
        /// Parses net output (detect) to predictions.
        /// </summary>
        private List<YoloPrediction> ParseDetect(DenseTensor<float> output, Image image)
        {
            // 不可改变的有序集合
            var result = ImmutableSortedSet.Create<YoloPrediction>();

            var (w, h) = (image.Width, image.Height); // image w and h
            var (xGain, yGain) = (_model.Width / (float)w, _model.Height / (float)h); // x, y gains
            var gain = Math.Min(xGain, yGain); // gain = resized / original

            var (xPad, yPad) = ((_model.Width - w * gain) / 2, (_model.Height - h * gain) / 2); // left, right pads

            Parallel.For(0, (int)output.Length / _model.Dimensions, (i) =>
            {
                if (output[0, i, 4] <= _model.Confidence) return; // skip low obj_conf results

                for (int j = 5; j < _model.Dimensions; j++)
                {
                    output[0, i, j] = output[0, i, j] * output[0, i, 4]; // mul_conf = obj_conf * cls_conf

                    if (output[0, i, j] <= _model.MulConfidence) continue; // skip low mul_conf results

                    float xMin = ((output[0, i, 0] - output[0, i, 2] / 2) - xPad) / gain; // unpad bbox tlx to original
                    float yMin = ((output[0, i, 1] - output[0, i, 3] / 2) - yPad) / gain; // unpad bbox tly to original
                    float xMax = ((output[0, i, 0] + output[0, i, 2] / 2) - xPad) / gain; // unpad bbox brx to original
                    float yMax = ((output[0, i, 1] + output[0, i, 3] / 2) - yPad) / gain; // unpad bbox bry to original

                    xMin = Clamp(xMin, 0, w - 0); // clip bbox tlx to boundaries
                    yMin = Clamp(yMin, 0, h - 0); // clip bbox tly to boundaries
                    xMax = Clamp(xMax, 0, w - 1); // clip bbox brx to boundaries
                    yMax = Clamp(yMax, 0, h - 1); // clip bbox bry to boundaries

                    YoloLabel label = _model.Labels[j - 5];

                    var prediction = new YoloPrediction(label, output[0, i, j])
                    {
                        Rectangle = new RectangleF(xMin, yMin, xMax - xMin, yMax - yMin)
                    };

                    // 先转为builder，提高效率
                    var builder = result.ToBuilder();
                    // 添加预测结果
                    builder.Add(prediction);
                    // 转为不可变集合
                    result = builder.ToImmutable();
                }
            });

            // 返回前MaxDetections个结果
            return result.Take(model.MaxDetections).ToList();
        }

        /// <summary>
        /// Parses net outputs (sigmoid) to predictions.
        /// </summary>
        private List<YoloPrediction> ParseSigmoid(DenseTensor<float>[] output, Image image)
        {
            var result = new ConcurrentBag<YoloPrediction>();

            var (w, h) = (image.Width, image.Height); // image w and h
            var (xGain, yGain) = (_model.Width / (float)w, _model.Height / (float)h); // x, y gains
            var gain = Math.Min(xGain, yGain); // gain = resized / original

            var (xPad, yPad) = ((_model.Width - w * gain) / 2, (_model.Height - h * gain) / 2); // left, right pads

            Parallel.For(0, output.Length, (i) => // iterate model outputs
            {
                int shapes = _model.Shapes[i]; // shapes per output

                for (int a = 0; a < _model.Anchors[0].Length; a++)
                {
                    for (int y = 0; y < shapes; y++)
                    {
                        for (int x = 0; x < shapes; x++)
                        {
                            int offset = (shapes * shapes * a + shapes * y + x) * _model.Dimensions;

                            float[] buffer = output[i].Skip(offset).Take(_model.Dimensions).Select(Sigmoid).ToArray();

                            if (buffer[4] <= _model.Confidence) return; // skip low obj_conf results

                            List<float> scores = buffer.Skip(5).Select(b => b * buffer[4]).ToList(); // mul_conf = obj_conf * cls_conf

                            float mulConfidence = scores.Max(); // max confidence score

                            if (mulConfidence <= _model.MulConfidence) return; // skip low mul_conf results

                            float rawX = (buffer[0] * 2 - 0.5f + x) * _model.Strides[i]; // predicted bbox x (center)
                            float rawY = (buffer[1] * 2 - 0.5f + y) * _model.Strides[i]; // predicted bbox y (center)

                            float rawW = (float)Math.Pow(buffer[2] * 2, 2) * _model.Anchors[i][a][0]; // predicted bbox w
                            float rawH = (float)Math.Pow(buffer[3] * 2, 2) * _model.Anchors[i][a][1]; // predicted bbox h

                            float[] xyxy = Xywh2xyxy(new float[] { rawX, rawY, rawW, rawH });

                            float xMin = Clamp((xyxy[0] - xPad) / gain, 0, w - 0); // unpad, clip tlx
                            float yMin = Clamp((xyxy[1] - yPad) / gain, 0, h - 0); // unpad, clip tly
                            float xMax = Clamp((xyxy[2] - xPad) / gain, 0, w - 1); // unpad, clip brx
                            float yMax = Clamp((xyxy[3] - yPad) / gain, 0, h - 1); // unpad, clip bry

                            YoloLabel label = _model.Labels[scores.IndexOf(mulConfidence)];

                            var prediction = new YoloPrediction(label, mulConfidence)
                            {
                                Rectangle = new RectangleF(xMin, yMin, xMax - xMin, yMax - yMin)
                            };

                            result.Add(prediction);
                        }
                    }
                }
            });

            return result.ToList();
        }

        /// <summary>
        /// Parses net outputs (sigmoid or detect layer) to predictions.
        /// </summary>
        private List<YoloPrediction> ParseOutput(DenseTensor<float>[] output, Image image)
        {
            //stopwatch.Restart();
            List<YoloPrediction> predictions = _model.UseDetect ? ParseDetect(output[0], image) : ParseSigmoid(output, image);
            //Console.WriteLine("output:" + stopwatch.ElapsedMilliseconds);
            return predictions;
        }

        /// <summary>
        /// Removes overlaped duplicates (nms).
        /// </summary>
        private List<YoloPrediction> Supress(List<YoloPrediction> items)
        {
            var result = new List<YoloPrediction>(items);

            foreach (var item in items) // iterate every prediction
            {
                foreach (var current in result.ToList()) // make a copy for each iteration
                {
                    if (current == item) continue;

                    var (rect1, rect2) = (item.Rectangle, current.Rectangle);

                    RectangleF intersection = RectangleF.Intersect(rect1, rect2);

                    float intArea = intersection.Area(); // intersection area
                    float unionArea = rect1.Area() + rect2.Area() - intArea; // union area
                    float overlap = intArea / unionArea; // overlap ratio

                    if (overlap >= _model.Overlap)
                    {
                        if (item.Score >= current.Score)
                        {
                            result.Remove(current);
                        }
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// Runs object detection.
        /// </summary>
        public List<YoloPrediction> Predict(Image image)
        {
            return Supress(ParseOutput(Inference(image), image));
        }

        /// <summary>
        /// Creates new instance of YoloScorer with weights path and options.
        /// </summary>
        public YoloScorer(YoloModel yoloModel, string weights, SessionOptions opts = null)
        {
            oclCaller = new OclCaller();
            oclCaller.Init();
            Stopwatch sw = new Stopwatch();
            sw.Start();
            _inferenceSession = new InferenceSession(File.ReadAllBytes(weights), opts ?? new SessionOptions());
            sw.Stop();
            Debug.WriteLine(sw.ElapsedMilliseconds);

            AddLabels(yoloModel);
            _model = yoloModel;
        }

        private void AddLabels(YoloModel yoloModel)
        {
            // Labels 列表没有标签
            if (yoloModel.Labels.Count == 0)
            {
                NodeMetadata data = _inferenceSession.OutputMetadata.Values.First();
                // 获取
                if (data.Dimensions.Length == 3)
                {
                    // 类别数量      data.Dimension：1*25200*(classCount+5)
                    int classCount = data.Dimensions[2] - 5;
                    for (int i = 0; i < classCount; i++)
                    {
                        YoloLabel yoloLabel = new YoloLabel(i, "Class" + i, YoloLabel.Colors[i % YoloLabel.Colors.Length]);
                        yoloModel.Labels.Add(yoloLabel);
                    }
                }
                else
                {
                    Debug.WriteLine("获取onnx输出元数据失败。");
                }
            }
        }

        /// <summary>
        /// Creates new instance of YoloScorer with weights stream and options.
        /// </summary>
        public YoloScorer(YoloModel yoloModel, Stream weights, SessionOptions opts = null)
        {
            oclCaller = new OclCaller();
            oclCaller.Init();

            using (var reader = new BinaryReader(weights))
            {
                _inferenceSession = new InferenceSession(reader.ReadBytes((int)weights.Length), opts ?? new SessionOptions());
            }
            AddLabels(yoloModel);
        }

        /// <summary>
        /// Creates new instance of YoloScorer with weights bytes and options.
        /// </summary>
        public YoloScorer(YoloModel yoloModel, byte[] weights, SessionOptions opts = null)
        {
            oclCaller = new OclCaller();
            oclCaller.Init();

            _inferenceSession = new InferenceSession(weights, opts ?? new SessionOptions());
            AddLabels(yoloModel);
        }

        /// <summary>
        /// Disposes YoloScorer instance.
        /// </summary>
        public void Dispose()
        {
            _inferenceSession.Dispose();
            oclCaller.Dispose();
        }
    }


    internal class OclCaller : IDisposable
    {
        private const string clCode = @"
            __kernel void img_to_tensor(__global int* src_a, __global float* res)
            {
                const int idx = get_global_id(0);
                res[idx] = src_a[idx]/255.0f;
            }
            ";

        private Context context;
        private CommandQueue commandQueue;
        private Dictionary<string, Kernel> kernels;
        //TODO 640*640*3的图像，计算下来用9ms左右。必须要有GPU（可以不是英伟达）才进行运算。多GPU或CPU+GPU运算应该可以更快。
        internal void Init()
        {
            OpenCL.Net.ErrorCode errorCode;
            Platform[] platforms = Cl.GetPlatformIDs(out errorCode);
            CheckErrorCode(errorCode);

            List<Device> devicesList = new List<Device>();
            foreach (Platform platform in platforms)
            {
                string platformName = Cl.GetPlatformInfo(platform, OpenCL.Net.PlatformInfo.Name, out errorCode).ToString();
                CheckErrorCode(errorCode);

                Console.WriteLine("Platform: " + platformName);
                //We will be looking only for GPU devices
                foreach (OpenCL.Net.Device device in Cl.GetDeviceIDs(platform, OpenCL.Net.DeviceType.Gpu, out errorCode))
                {
                    CheckErrorCode(errorCode);

                    Console.WriteLine("Device: " + device.ToString());
                    devicesList.Add(device);
                }
            }
            if (devicesList.Count <= 0)
            {
                throw new Exception("No devices found.");
            }
            //选中运算设备,这里选第一个其它的释放掉
            var oclDevice = devicesList[0];

            //根据配置建立上下文
            context = Cl.CreateContext(null, 1, new[] { oclDevice }, null, IntPtr.Zero, out errorCode);
            CheckErrorCode(errorCode);

            commandQueue = Cl.CreateCommandQueue(context, oclDevice, CommandQueueProperties.OutOfOrderExecModeEnable, out errorCode);
            CheckErrorCode(errorCode);

            //定义一个字典用来存放所有核
            kernels = new Dictionary<string, Kernel>();
            using (var program = Cl.CreateProgramWithSource(context, 1, new[] { clCode }, null, out errorCode))
            {
                CheckErrorCode(errorCode);

                errorCode = Cl.BuildProgram(program, 1, new[] { oclDevice }, "", null, IntPtr.Zero);
                CheckErrorCode(errorCode);

                foreach (var item in new[] { "img_to_tensor" })
                {
                    kernels.Add(item, Cl.CreateKernel(program, item, out errorCode));
                    CheckErrorCode(errorCode);
                }
            }
        }

        internal void Compute(int[] a, out float[] output)
        {
            output = new float[a.Length];

            OpenCL.Net.ErrorCode errorCode;

            Event eventExecutedVectorAddGpu;
            Event eventExecutedReadResult;
            IMem<int> n1 = null;
            IMem<float> n3 = null;
            try
            {
                #region 创建并传参
                //在显存创建缓冲区并把HOST的数据拷贝过去
                n1 = Cl.CreateBuffer(context, MemFlags.CopyHostPtr, a, out errorCode);
                CheckErrorCode(errorCode);
                //还有一个缓冲区用来接收回参
                n3 = Cl.CreateBuffer(context, MemFlags.CopyHostPtr, output, out errorCode);
                CheckErrorCode(errorCode);

                //把参数填进Kernel里
                errorCode = Cl.SetKernelArg(kernels["img_to_tensor"], 0, n1);
                CheckErrorCode(errorCode);
                errorCode = Cl.SetKernelArg(kernels["img_to_tensor"], 1, n3);
                CheckErrorCode(errorCode);
                #endregion

                //把调用请求添加到队列里,参数分别是:Kernel,数据的维度,每个维度的全局工作项ID偏移,每个维度工作项数量(我们这里有4个数据,所以设为4),每个维度的工作组长度(这里设为每4个一组)
                errorCode = Cl.EnqueueNDRangeKernel(commandQueue, kernels["img_to_tensor"], 1, null, new[] { (IntPtr)(a.Length) }, new[] { (IntPtr)1 }, 0, null, out eventExecutedVectorAddGpu);
                CheckErrorCode(errorCode);
                //添加一个读取数据命令到队列里,用来读取运算结果
                errorCode = Cl.EnqueueReadBuffer(commandQueue, n3, Bool.True, output, 1, new[] { eventExecutedVectorAddGpu }, out eventExecutedReadResult);
                CheckErrorCode(errorCode);
                //开始执行
                errorCode = Cl.Finish(commandQueue);
                CheckErrorCode(errorCode);

                eventExecutedVectorAddGpu.Dispose();
                eventExecutedReadResult.Dispose();
            }
            finally
            {
                if (n1 != null)
                {
                    n1.Dispose();
                }
                if (n3 != null)
                {
                    n3.Dispose();
                }
            }
        }

        private static void CheckErrorCode(OpenCL.Net.ErrorCode errorCode)
        {
            if (errorCode != OpenCL.Net.ErrorCode.Success)
            {
                throw new Exception(errorCode.ToString());
            }
        }

        internal void Dispose()
        {
            context.Dispose();
            commandQueue.Dispose();
            foreach (var k in kernels.Values)
            {
                k.Dispose();
            }
        }

        void IDisposable.Dispose()
        {
            this.Dispose();
        }
    }
}
