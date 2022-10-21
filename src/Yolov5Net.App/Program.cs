using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using Yolov5Net.Scorer;

namespace Yolov5Net.App
{
    class Program
    {
        public static object EasyCl { get; private set; }

        static void Main(string[] args)
        {
            // 单张测试
            PredictTest();
            // 性能测试
            BenchMark();
        }
        private static void PredictTest()
        {
            using var image = Image.FromFile("Assets/guangui_2554.jpg");

            using var scorer = new YoloScorer<TrackModel>("Assets/Weights/track_n.onnx");

            List<YoloPrediction> predictions = scorer.Predict(image);

            using var graphics = Graphics.FromImage(image);

            foreach (var prediction in predictions) // iterate predictions to draw results
            {
                double score = Math.Round(prediction.Score, 2);

                graphics.DrawRectangles(new Pen(prediction.Label.Color, 1),
                    new[] { prediction.Rectangle });

                var (x, y) = (prediction.Rectangle.X - 3, prediction.Rectangle.Y - 23);

                graphics.DrawString($"{prediction.Label.Name} ({score})",
                    new Font("Consolas", 16, GraphicsUnit.Pixel), new SolidBrush(prediction.Label.Color),
                    new PointF(x, y));
            }

            image.Save("Assets/result.jpg");
        }

        private static void BenchMark()
        {
            var sw = new Stopwatch();
            SessionOptions sessionOptions = new SessionOptions();
            sessionOptions.AppendExecutionProvider_CUDA();
            using (var image = Image.FromFile("Assets/guangui_2554.jpg"))
            using (var scorer = new YoloScorer<TrackModel>("Assets/Weights/track_n.onnx", sessionOptions))
            {
                List<long> stats = new List<long>();

                for (int i = 0; i < 100; i++)
                {
                    sw.Restart();
                    scorer.Predict(image);
                    long fps = 1000 / sw.ElapsedMilliseconds;
                    stats.Add(fps);
                    sw.Stop();
                }

                stats.Sort();
                Console.WriteLine($@"
                    Max FPS: {stats[stats.Count - 1]}
                    Avg FPS: {Avg(stats)}
                    Min FPS: {stats[0]}
                ");
            }
        }
        private static int Avg(List<long> stats)
        {
            long sum = 0;
            foreach (long i in stats)
            {
                sum += i;
            }
            return (int)(sum / stats.Count);
        }
    }
}
