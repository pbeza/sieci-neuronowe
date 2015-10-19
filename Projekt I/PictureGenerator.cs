namespace sieci_neuronowe
{
    #region

    using System.Collections.Generic;
    using System.Drawing;
    using System.Drawing.Imaging;
    using System.Linq;
    using System.Runtime.InteropServices;

    using Encog.ML;
    using Encog.ML.Data;
    using Encog.ML.Data.Basic;
    using Encog.ML.Data.Versatile;

    #endregion

    public static class PictureGenerator
    {
        #region Public Methods and Operators

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
            BitmapData lck = bmp.LockBits(
                new Rectangle(0, 0, resolutionX, resolutionY), 
                ImageLockMode.WriteOnly, 
                PixelFormat.Format32bppArgb);
            const double ResolutionMult = 3.0;
            const double CoordOffset = -1.5;
            double stepX = ResolutionMult / resolutionX;
            double stepY = ResolutionMult / resolutionY;
            for (int i = 0; i < resolutionX; i++)
            {
                double x = (i * stepX) + CoordOffset;
                for (int j = 0; j < resolutionY; j++)
                {
                    double y = (j * stepY) + CoordOffset;
                    IMLData output = testFunction.Compute(new BasicMLData(new[] { x, y }));
                    string stringChosen = helper.DenormalizeOutputVectorToString(output)[0];
                    int result = int.Parse(stringChosen);
                    int colorRGB = RGBFromInt(result, points.Any());
                    Marshal.WriteInt32(lck.Scan0 + (((i * lck.Width) + j) * 4), colorRGB);
                }
            }

            foreach (var pt in points)
            {
                if (pt.Correct < 0)
                {
                    continue;
                }

                double x = pt.X;
                double y = pt.Y;
                int colorRGB = RGBFromInt(pt.Correct, false);

                var i = (int)((x - CoordOffset) / stepX);
                var j = (int)((y - CoordOffset) / stepY);
                if (i > 0 && i < lck.Width && j > 0 && j < lck.Height)
                {
                    Marshal.WriteInt32(lck.Scan0 + (((i * lck.Width) + j) * 4), colorRGB);
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
            BitmapData lck = bmp.LockBits(
                new Rectangle(0, 0, resolutionX, resolutionY), 
                ImageLockMode.WriteOnly, 
                PixelFormat.Format32bppArgb);
            var xmin = points.Min(p => p.X);
            var xmax = points.Max(p => p.X);
            var ymin = points.Min(p => p.Y);
            var ymax = points.Max(p => p.Y);
            const double Padding = 1.5;
            double dx = xmax - xmin;
            double dy = ymax - ymin;
            double resolutionMultX = Padding * dx;
            double resolutionMultY = Padding * dy;
            double stepX = resolutionMultX / resolutionX;
            double stepY = resolutionMultY / resolutionY;

            double coordOffsetX = xmin + (dx / 10);
            double coordOffsetY = ymin + (dy / 10);

            for (int i = 0; i < resolutionX; i++)
            {
                double x = (i * stepX) + coordOffsetX;
                IMLData output = testFunction.Compute(new BasicMLData(new[] { x }));
                string stringChosen = helper.DenormalizeOutputVectorToString(output)[0];
                double y = double.Parse(stringChosen);
                int colorRGB = RGBFromInt(1, false);
                var j = (int)((y - coordOffsetY) / stepY);
                if (i > 0 && i < lck.Width && j > 0 && j < lck.Height)
                {
                    Marshal.WriteInt32(lck.Scan0 + (((i * lck.Width) + j) * 4), colorRGB);
                }
            }

            foreach (var pt in points)
            {
                if (pt.Y < 0)
                {
                    continue;
                }

                double x = pt.X;
                double y = pt.Y;
                int colorRGB = RGBFromInt(2, false);

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

        #endregion

        #region Methods

        private static int RGBFromInt(int i, bool lowIntensity)
        {
            bool red = (i % 4) == 1;
            bool green = (i % 4) == 2;
            bool blue = (i % 4) == 3;
            int val = lowIntensity ? 63 : 255;
            int r = red ? val : 0;
            int g = green ? val : 0;
            int b = blue ? val : 0;
            return (0x0ff << 24) | ((r & 0x0ff) << 16) | ((g & 0x0ff) << 8) | ((b & 0x0ff) << 0);
        }

        #endregion
    }
}