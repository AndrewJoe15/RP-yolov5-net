using System;
using System.Collections.Generic;
using System.Drawing;

namespace Yolov5Net.Scorer
{
    /// <summary>
    /// Object prediction.
    /// </summary>
    public class YoloPrediction : IComparable, IComparer<YoloPrediction>
    {
        public YoloLabel Label { get; set; }
        public RectangleF Rectangle { get; set; }
        public float Score { get; set; }

        public YoloPrediction() { }

        public YoloPrediction(YoloLabel label, float confidence) : this(label)
        {
            Score = confidence;
        }

        public YoloPrediction(YoloLabel label)
        {
            Label = label;
        }

        public int Compare(YoloPrediction x, YoloPrediction y)
        {
            if (x.Score == y.Score)
                return 0;

            return x.Score.CompareTo(y.Score);
        }

        public int CompareTo(object obj)
        {
            var y = obj as YoloPrediction;
            return Score.CompareTo(y.Score);
        }
    }
}
