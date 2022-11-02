using System.Drawing;

namespace Yolov5Net.Scorer
{
    /// <summary>
    /// Label of detected object.
    /// </summary>
    public class YoloLabel
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public YoloLabelKind Kind { get; set; }
        public Color Color { get; set; }

        public YoloLabel()
        {
            Color = Color.Yellow;
        }

        public YoloLabel(int id, string name, Color color)
        {
            Id = id;
            Name = name;
            Color = color;
        }

        public static Color[] Colors { get; } = new Color[]
        {
            Color.Green,
            Color.Red,
            Color.Blue,
            Color.Yellow,
            Color.Lime,
            Color.Cyan,
            Color.Magenta,
            Color.Brown,
            Color.DarkGreen,
            Color.DarkRed,
            Color.DarkGray,
            Color.LightCoral,
            Color.LightSalmon,
            Color.Maroon
        };
    }
}
