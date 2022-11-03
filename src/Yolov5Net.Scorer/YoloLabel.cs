using System.Drawing;
using System.Xml.Serialization;

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
        [XmlIgnore()] //Color对象没法序列化
        public Color Color { get; set; }

        // 用于序列化Color
        [XmlElement(nameof(Color))]
        public string XmlColor
        {
            get => Color.Name;
            set => Color = Color.FromName(value);
        }

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
