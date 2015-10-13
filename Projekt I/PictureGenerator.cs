namespace sieci_neuronowe
{
    #region

    using System.Drawing;
    using System.Drawing.Imaging;
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
        ///     Testuje tylko punkty z zakresu 0.0-1.0.
        /// </summary>
        /// <param name="path">Ścieżka do pliku</param>
        /// <param name="testFunction">Funkcja którą będziemy testować punkty</param>
        /// <param name="dataSet">Zbiór punktów do narysowania (może być null)</param>
        /// <param name="helper">Helper do normalizacji</param>
        /// <param name="resolutionX">Rozdzielczość w x</param>
        /// <param name="resolutionY">j.w, dla y</param>
        public static void DrawArea(
            string path, 
            IMLRegression testFunction, 
            MatrixMLDataSet dataSet, 
            NormalizationHelper helper, 
            int resolutionX, 
            int resolutionY)
        {
            var bmp = new Bitmap(resolutionX, resolutionY);
            BitmapData lck = bmp.LockBits(
                new Rectangle(0, 0, resolutionX, resolutionY), 
                ImageLockMode.WriteOnly, 
                PixelFormat.Format32bppArgb);
            double stepX = 1.0 / resolutionX;
            double stepY = 1.0 / resolutionY;
            for (int i = 0; i < resolutionX; i++)
            {
                double x = i * stepX;
                for (int j = 0; j < resolutionY; j++)
                {
                    double y = j * stepY;
                    IMLData output = testFunction.Compute(new BasicMLData(new[] { x, y }));
                    string[] denormalized = helper.DenormalizeOutputVectorToString(output);
                    int result = int.Parse(denormalized[0]);
                    int colorRGB = rgbFromInt(result, dataSet != null);
                    Marshal.WriteInt32(lck.Scan0 + (((i * lck.Width) + j) * 4), colorRGB);
                }
            }

            if (dataSet != null)
            {
                for (int k = 0; k < dataSet.Count; k++)
                {
                    double x = dataSet.Data[k][0];
                    double y = dataSet.Data[k][1];
                    var dummy = new BasicMLData(new[] { x, y, dataSet.Data[k][2] });
                    string[] denormalized = helper.DenormalizeOutputVectorToString(dummy);
                    int result = int.Parse(denormalized[0]);
                    int colorRGB = rgbFromInt(result, false);
                    var i = (int)(x * resolutionX);
                    var j = (int)(y * resolutionY);
                    if (i > 0 && i < lck.Width && j > 0 && j < lck.Height)
                    {
                        Marshal.WriteInt32(lck.Scan0 + (((i * lck.Width) + j) * 4), colorRGB);
                    }
                }
            }

            bmp.UnlockBits(lck);
            bmp.Save(path);
        }

        #endregion

        #region Methods

        private static int rgbFromInt(int i, bool lowIntensity)
        {
            bool red = (i & 2) != 0;
            bool green = (i & 3) != 0;
            bool blue = (i & 4) != 0;
            int val = lowIntensity ? 63 : 255;
            int r = red ? val : 0;
            int g = green ? val : 0;
            int b = blue ? val : 0;
            return (0x0ff << 24) | ((r & 0x0ff) << 16) | ((g & 0x0ff) << 8) | ((b & 0x0ff) << 0);
        }

        #endregion
    }
}