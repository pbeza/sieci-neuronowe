namespace sieci_neuronowe
{
    #region

    using System.Collections.Generic;
    using System.Drawing;
    using System.Drawing.Imaging;
    using System.Linq;
    using System.Runtime.InteropServices;

    using Encog.ML;
    using Encog.ML.Data.Basic;
    using Encog.ML.Data.Versatile;

    #endregion

    public static class PictureGenerator
    {
        #region Public Methods and Operators

        /// <summary>
        ///     Generuje obrazek przedstawiający jak sieć dzieli przestrzeń.
        ///     Testuje tylko punkty z zakresu 0.0-1.0.
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
            List<NeuroPoint> points, 
            NormalizationHelper helper, 
            int resolutionX, 
            int resolutionY)
        {
            var bmp = new Bitmap(resolutionX, resolutionY);
            BitmapData lck = bmp.LockBits(
                new Rectangle(0, 0, resolutionX, resolutionY), 
                ImageLockMode.WriteOnly, 
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
                    IMLData output =
                        new BasicMLData(new[] { x, y, testFunction.Compute(new BasicMLData(new[] { x, y }))[0] });
                    string stringChosen = helper.DenormalizeOutputVectorToString(output)[0];
                    int result = int.Parse(stringChosen);
                    Marshal.WriteInt32(lck.Scan0 + (((i * lck.Width) + j) * 4), colorRGB);
                }
            }

            foreach (var pt in points)
            {
                var x = pt.X;
                var y = pt.Y;
                int colorRGB;
                if (pt.Correct >= 0)
                {
                    colorRGB = RGBFromInt(pt.Correct, false);
                }
                else
                {
                    // Czarny punkt
                    colorRGB = 0x0ff << 24;
                }

                var i = (int)((x - coordOffset) / stepX);
                var j = (int)((y - coordOffset) / stepY);
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
            var red = (i % 4) == 1;
            var green = (i % 4) == 2;
            var blue = (i % 4) == 3;
            var val = lowIntensity ? 63 : 255;
            var r = red ? val : 0;
            var g = green ? val : 0;
            var b = blue ? val : 0;
            return (0x0ff << 24) | ((r & 0x0ff) << 16) | ((g & 0x0ff) << 8) | ((b & 0x0ff) << 0);
        }

        #endregion
    }
}