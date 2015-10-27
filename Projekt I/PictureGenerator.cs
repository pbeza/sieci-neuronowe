﻿namespace sieci_neuronowe
{
    using System;
    using System.Collections.Generic;
    using System.Drawing;
    using System.Drawing.Imaging;
    using System.Globalization;
    using System.Linq;
    using System.Runtime.InteropServices;

    using Encog.ML;
    using Encog.ML.Data.Basic;
    using Encog.ML.Data.Versatile;

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
            int categoryCount = helper.OutputColumns.Count;
            bool dim = points.Any();
            for (var i = 0; i < resolutionX; i++)
            {
                var x = (i * stepX) + coordOffset;
                for (var j = 0; j < resolutionY; j++)
                {
                    var y = (j * stepY) + coordOffset;
                    BasicMLData arr = new BasicMLData(2);
                    arr[0] = x;
                    arr[1] = y;
                    var strings = new[] { x.ToString(CultureInfo.InvariantCulture), y.ToString(CultureInfo.InvariantCulture) };
                    helper.NormalizeInputVector(strings, arr.Data, false);
                    var output = testFunction.Compute(arr);
                    var denormalizedOutput = helper.DenormalizeOutputVectorToString(output);
                    //var result = NeuralNetwork.ActualCategory(output);
                    var result = int.Parse(denormalizedOutput[0]);
                    float colorMul = dim ? 0.25f : 1;
                    if (result < 0)
                    {
                        float degree;
                        result = NeuralNetwork.ClosestCategory(output, out degree);
                        colorMul = degree / 2;
                    }
                    
                    var colorRGB = RGBFromInt(result, colorMul);
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
                var colorRGB = RGBFromInt((int)pt.Correct);

                var i = (int)((y - coordOffset) / stepY);
                var j = (int)((x - coordOffset) / stepX);
                if (i >= 0 && i < lck.Height && j >= 0 && j < lck.Width)
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
            List<ClassifiedPoint> known = points.FindAll(p => Math.Abs(p.Correct) > 0.000001);
            var xmin = known.Min(p => p.X);
            var xmax = known.Max(p => p.X);
            var ymin = known.Min(p => p.Correct);
            var ymax = known.Max(p => p.Correct);
            const double padding = 1.5;
            double dx = xmax - xmin;
            double dy = ymax - ymin;
            double resolutionMultX = padding * dx;
            double resolutionMultY = padding * dy;
            double stepX = resolutionMultX / resolutionX;
            double stepY = resolutionMultY / resolutionY;
            double centerX = (xmax + xmin) / 2;
            double centerY = (ymax + ymin) / 2;
            double toRight = padding * (xmax - centerX);
            double toBottom = padding * (ymax - centerY);
            double coordOffsetX = centerX - toRight;
            double coordOffsetY = centerY - toBottom;

            var testX1 = (xmin - coordOffsetX) / stepX;
            var testX2 = (xmax - coordOffsetX) / stepX;
            var testY1 = (ymin - coordOffsetY) / stepY;
            var testY2 = (ymax - coordOffsetY) / stepY;

            for (var j = 0; j < resolutionX; j++)
            {
                var x = j * stepX + coordOffsetX;
                var output = testFunction.Compute(new BasicMLData(new[] { x }));
                var stringChosen = helper.DenormalizeOutputVectorToString(output)[0];
                var y = double.Parse(stringChosen);
                var colorRGB = RGBFromInt(1);
                var i = (int)((y - coordOffsetY) / stepY);
                if (i >= 0 && i < lck.Height && j >= 0 && j < lck.Width)
                {
                    Marshal.WriteInt32(lck.Scan0 + (i * lck.Width + j) * 4, colorRGB);
                }
            }

            bmp.UnlockBits(lck);
            if (!known.Any())
            {
                bmp.Save(path);
                return;
            }

            known.Sort((left, right) => (left.X < right.X ? (right.X > left.X ? 1 : 0 ) : -1));
            var g = Graphics.FromImage(bmp);
            var pen = Pens.Green;

            int iLast = (int)((known[0].Correct - coordOffsetY) / stepY);
            int jLast = (int)((known[0].X - coordOffsetX) / stepX);
            foreach (var pt in known)
            {
                var x = pt.X;
                var y = pt.Correct;

                var i = (int)((y - coordOffsetY) / stepY);
                var j = (int)((x - coordOffsetX) / stepX);
                g.DrawLine(pen, jLast, iLast, j, i);
                iLast = i;
                jLast = j;
            }

            bmp.Save(path);
        }

        private static readonly Dictionary<int, Tuple<int, int, int>> colorMap =
            new Dictionary<int, Tuple<int, int, int>>
                {
                    { -2, Tuple.Create(0, 0, 0) },
                    { -1, Tuple.Create(0, 0, 0) },
                    { 0, Tuple.Create(0, 0, 0) },
                    { 1, Tuple.Create(255, 0, 0) },
                    { 2, Tuple.Create(0, 255, 0) },
                    { 3, Tuple.Create(0, 0, 255) },
                    { 4, Tuple.Create(255, 255, 0) },
                    { 5, Tuple.Create(0, 255, 255) },
                    { 6, Tuple.Create(255, 0, 255) },
                    { 7, Tuple.Create(255, 255, 255) },
                };

        private static int RGBFromInt(int i, float mul = 1.0f)
        {
            Tuple<int, int, int> tp;
            var got = colorMap.TryGetValue(i, out tp);
            if (!got)
            {
                tp = new Tuple<int, int, int>(255, 255, 255);
            }

            int r = (int)(tp.Item1 * mul);
            int g = (int)(tp.Item2 * mul);
            int b = (int)(tp.Item3 * mul);
            return (0x0ff << 24) | ((r & 0x0ff) << 16) | ((g & 0x0ff) << 8) | ((b & 0x0ff) << 0);
        }
    }
}