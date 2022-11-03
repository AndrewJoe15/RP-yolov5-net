using System.Collections.Generic;
using System.ComponentModel;
using System.Xml.Serialization;

namespace Yolov5Net.Scorer
{
    /// <summary>
    /// Model descriptor.
    /// </summary>
    public class YoloModel : INotifyPropertyChanged
    {
        public int Width { get; set; } = 640;
        public int Height { get; set; } = 640;
        public int Depth { get; set; } = 3;

        public int Dimensions { get => Labels.Count + 5;}

        public int[] Strides { get; set; } = new int[] { 8, 16, 32 };
        public int[][][] Anchors { get; set; } = new int[][][]
            {
                new int[][] { new int[] { 010, 13 }, new int[] { 016, 030 }, new int[] { 033, 023 } },
                new int[][] { new int[] { 030, 61 }, new int[] { 062, 045 }, new int[] { 059, 119 } },
                new int[][] { new int[] { 116, 90 }, new int[] { 156, 198 }, new int[] { 373, 326 } }
            };
        public int[] Shapes { get; set; } = new int[] { 80, 40, 20 };

        public float Confidence
        {
            get => _Confidence;
            set
            {
                if (Confidence != value)
                {
                    _Confidence = value;
                    OnPropertyChanged(nameof(Confidence));
                }
            }
        }
        public float MulConfidence
        {
            get => _MulConfidence;
            set
            {
                if (MulConfidence != value)
                {
                    _MulConfidence = value;
                    OnPropertyChanged(nameof(MulConfidence));
                }
            }
        }
        public float Overlap
        {
            get => _Overlap;
            set
            {
                if (Overlap != value)
                {
                    _Overlap = value;
                    OnPropertyChanged(nameof(Overlap));
                }
            }
        }

        public int MaxDetections
        {
            get => _MaxDetections;
            set
            {
                if (MaxDetections != value)
                {
                    _MaxDetections = value;
                    OnPropertyChanged(nameof(MaxDetections));
                }
            }
        }

        public string[] Outputs { get; set; } = new[] { "output" };
        public List<YoloLabel> Labels { get; set; } = new List<YoloLabel>();
        public bool UseDetect { get; set; } = true;


        public event PropertyChangedEventHandler PropertyChanged;
        private void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        private float _Confidence;
        private float _MulConfidence;
        private float _Overlap;
        private int _MaxDetections;
    }
}
