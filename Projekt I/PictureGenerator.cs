using Encog.ML;
using Encog.ML.Data.Basic;
using Encog.ML.Data.Versatile;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Runtime.InteropServices;

namespace sieci_neuronowe
{
    public static class PictureGenerator
    {
        /// <summary>
        ///     Generuje obrazek przedstawiający jak sieć dzieli przestrzeń.
        ///     Testuje tylko punkty z zakresu 0.0 - 1.0.
        /// </summary>
        /// <param name="path">Ścieżka do pliku</param>
        /// <param name="testFunction">Funkcja którą będziemy testować punkty</param>
        /// <param name="points">Zbiór zbiorów punktów do narysowania (może być pusty)</param>
        /// <param name="helper">Helper do normalizacji</param>
        /// <param name="resolutionX">Rozdzielczość w x</param>
        /// <param name="resolutionY">j.w, dla y</param>
        public static void DrawArea(
            string path,
            IMLRegression testFunction,
            List<ClassifiedPoint> points,
            NormalizationHelper helper,
            int resolutionX,
            int resolutionY)
        {
            var bmp = new Bitmap(resolutionX, resolutionY);
            var lck = bmp.LockBits(new Rectangle(0, 0, resolutionX, resolutionY), ImageLockMode.WriteOnly,
                PixelFormat.Format32bppArgb);
            const double resolutionMult = 3.0;
            const double coordOffset = -1.5;
            var stepX = resolutionMult / resolutionX;
            var stepY = resolutionMult / resolutionY;
            for (var i = 0; i < resolutionX; i++)
            {
                var x = (i * stepX) + coordOffset;
                for (var j = 0; j < resolutionY; j++)
                {
                    var y = (j * stepY) + coordOffset;
                    var output = testFunction.Compute(new BasicMLData(new[] { x, y }));
                    var stringChosen = helper.DenormalizeOutputVectorToString(output)[0];
                    var result = int.Parse(stringChosen);
                    var colorRGB = RGBFromInt(result, points.Any());
                    Marshal.WriteInt32(lck.Scan0 + (i * lck.Width + j) * 4, colorRGB);
                }
            }

            foreach (var pt in points)
            {
                if (pt.Correct < 0)
                {
                    continue;
                }

                var x = pt.X;
                var y = pt.Y;
                var colorRGB = RGBFromInt(pt.Correct, false);

                var i = (int)((x - coordOffset) / stepX);
                var j = (int)((y - coordOffset) / stepY);
                if (i > 0 && i < lck.Width && j > 0 && j < lck.Height)
                {
                    Marshal.WriteInt32(lck.Scan0 + (i * lck.Width + j) * 4, colorRGB);
                }
            }

            bmp.UnlockBits(lck);
            bmp.Save(path);
        }

        public static void DrawGraph(
            string path,
            IMLRegression testFunction,
            List<ClassifiedPoint> points,
            NormalizationHelper helper,
            int resolutionX,
            int resolutionY)
        {
            var bmp = new Bitmap(resolutionX, resolutionY);
            var lck = bmp.LockBits(new Rectangle(0, 0, resolutionX, resolutionY), ImageLockMode.WriteOnly,
                PixelFormat.Format32bppArgb);
            var xmin = points.Min(p => p.X);
            var xmax = points.Max(p => p.X);
            var ymin = points.Min(p => p.Y);
            var ymax = points.Max(p => p.Y);
            const double padding = 1.5;
            double dx = xmax - xmin,
                dy = ymax - ymin,
                resolutionMultX = padding * dx,
                resolutionMultY = padding * dy,
                stepX = resolutionMultX / resolutionX,
                stepY = resolutionMultY / resolutionY,
                coordOffsetX = xmin + dx / 10,
                coordOffsetY = ymin + dy / 10;
            for (var i = 0; i < resolutionX; i++)
            {
                var x = i * stepX + coordOffsetX;
                var output = testFunction.Compute(new BasicMLData(new[] { x }));
                var stringChosen = helper.DenormalizeOutputVectorToString(output)[0];
                var y = double.Parse(stringChosen);
                var colorRGB = RGBFromInt(1, false);
                var j = (int)((y - coordOffsetY) / stepY);
                if (i > 0 && i < lck.Width && j > 0 && j < lck.Height)
                {
                    Marshal.WriteInt32(lck.Scan0 + (i * lck.Width + j) * 4, colorRGB);
                }
            }

            foreach (var pt in points)
            {
                if (pt.Y < 0)
                {
                    continue;
                }

                var x = pt.X;
                var y = pt.Y;
                var colorRGB = RGBFromInt(2, false);

                var i = (int)((x - coordOffsetX) / stepX);
                var j = (int)((y - coordOffsetY) / stepY);
                if (i > 0 && i < lck.Width && j > 0 && j < lck.Height)
                {
                    Marshal.WriteInt32(lck.Scan0 + (((i * lck.Width) + j) * 4), colorRGB);
                }
            }

            bmp.UnlockBits(lck);
            bmp.Save(path);
        }

        private static int RGBFromInt(int i, bool lowIntensity)
        {
            var red = (i % 4) == 1;
            var green = (i % 4) == 2;
            var blue = (i % 4) == 3;
            var val = lowIntensity ? 63 : 255;
            var r = red ? val : 0;
            var g = green ? val : 0;
            var b = blue ? val : 0;
            return (0x0ff << 24) | ((r & 0x0ff) << 16) | ((g & 0x0ff) << 8) | ((b & 0x0ff) << 0);
        }
    }
}