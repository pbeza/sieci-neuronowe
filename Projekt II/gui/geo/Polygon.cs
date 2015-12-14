namespace gui.geo
{
    #region

    using System;
    using System.Collections.Generic;
    using System.Drawing;
    using System.Linq;

    #endregion

    internal class Polygon
    {
        public Polygon(List<PointD> points)
        {
            Points = points;
        }

        public List<PointD> Points { get; private set; }

        private static double CrossProductLength(double Ax, double Ay, double Bx, double By, double Cx, double Cy)
        {
            var BAx = Ax - Bx;
            var BAy = Ay - By;
            var BCx = Cx - Bx;
            var BCy = Cy - By;

            return BAx * BCy - BAy * BCx;
        }

        private static double DotProduct(double Ax, double Ay, double Bx, double By, double Cx, double Cy)
        {
            var BAx = Ax - Bx;
            var BAy = Ay - By;
            var BCx = Cx - Bx;
            var BCy = Cy - By;

            return BAx * BCx + BAy * BCy;
        }

        private static double GetAngle(double Ax, double Ay, double Bx, double By, double Cx, double Cy)
        {
            var dot_product = DotProduct(Ax, Ay, Bx, By, Cx, Cy);
            var cross_product = CrossProductLength(Ax, Ay, Bx, By, Cx, Cy);
            return Math.Atan2(cross_product, dot_product);
        }

        public bool PointInPolygon(double X, double Y)
        {
            var lastPoint = Points.Count - 1;
            var totalAngle = GetAngle(Points[lastPoint].X, Points[lastPoint].Y, X, Y, Points[0].X, Points[0].Y);

            for (var i = 0; i < lastPoint; i++)
            {
                totalAngle += GetAngle(Points[i].X, Points[i].Y, X, Y, Points[i + 1].X, Points[i + 1].Y);
            }

            return Math.Abs(totalAngle) > 0.000001;
        }

        private double PolygonArea()
        {
            var lastPoint = Points.Count - 1;

            // Get the areas.
            double area = (Points[lastPoint].X - Points[0].X) * (Points[lastPoint].Y + Points[0].Y) / 2;
            for (int i = 0; i < Points.Count; i++)
            {
                area += (Points[i + 1].X - Points[i].X) * (Points[i + 1].Y + Points[i].Y) / 2;
            }

            // Return the result.
            return Math.Abs(area);
        }

        public void Draw(Point start, Point end, Transform transform, TerrainType[,] ret)
        {
            foreach (var p in Points)
            {
                var b = transform.ToBitmapCoords(p);
                ret[b.X, b.Y] = TerrainType.Building;
            }
            /*
            for (int x = start.X; x < end.X; x++)
            {
                double xLocal = x * step.X + bounds.MinLat;
                for (int y = start.Y; y < end.Y; y++)
                {
                    double yLocal = y * step.Y + bounds.MinLon;
                    if (poly.PointInPolygon(xLocal, yLocal))
                    {
                        ret[x, y] = TerrainType.Building;
                    }
                }
            }
             */
            int polyCorners = Points.Count;
            var pts = this.Points.Select(transform.ToBitmapCoordsD).ToArray();
            int drawn_pixels = 0;
            for (int pixelY = start.Y; pixelY < end.Y; pixelY++)
            {
                List<int> nodeX = new List<int>(pts.Length);
                int j = polyCorners - 1;
                for (int i = 0; i < polyCorners; i++)
                {
                    if ((pts[i].Y < pixelY && pts[j].Y > pixelY) || (pts[j].Y < pixelY && pts[i].Y > pixelY))
                    {
                        nodeX.Add((int)(pts[i].X + (pixelY - pts[i].Y) / (pts[j].Y - pts[i].Y) * (pts[j].X - pts[i].X)));
                    }
                    j = i;
                }

                if (nodeX.Count % 2 != 0)
                {
                    continue;
                }

                nodeX.Sort((a, b) => Math.Sign(a - b));

                for (int i = 0; i < nodeX.Count - 1; i += 2)
                {
                    if (nodeX[i] >= end.X)
                    {
                        break;
                    }
                    if (nodeX[i + 1] <= start.X)
                    {
                        continue;
                    }
                    nodeX[i] = Math.Max(nodeX[i], start.X);
                    nodeX[i + 1] = Math.Min(nodeX[i + 1], end.X);
                    for (int pixelX = nodeX[i]; pixelX < nodeX[i + 1]; pixelX++)
                    {
                        ret[pixelX, pixelY] = TerrainType.Building;
                    }
                    drawn_pixels += Math.Max(0, nodeX[i + 1] - nodeX[i]);
                }
            }
        }

        private static void Swap<T>(ref T a, ref T b)
        {
            T temp = a;
            a = b;
            b = temp;
        }

        private void DrawLine(PointD a, PointD b, Point start, Point end, TerrainType[,] ret)
        {
            float x1 = (float)a.X;
            float y1 = (float)a.Y;
            float x2 = (float)b.X;
            float y2 = (float)b.Y;
            bool steep = (Math.Abs(y2 - y1) > Math.Abs(x2 - x1));
            if (steep)
            {
                Swap(ref x1, ref y1);
                Swap(ref x2, ref y2);
            }

            if (x1 > x2)
            {
                Swap(ref x1, ref x2);
                Swap(ref y1, ref y2);
            }

            float dx = x2 - x1;
            float dy = Math.Abs(y2 - y1);

            float error = dx / 2.0f;
            int ystep = (y1 < y2) ? 1 : -1;
            int y = (int)y1;

            int maxX = (int)x2;

            for (int x = (int)x1; x < maxX; x++)
            {
                int finalX;
                int finalY;
                if (steep)
                {
                    finalX = y;
                    finalY = x;
                }
                else
                {
                    finalX = x;
                    finalY = y;
                }

                if (finalX >= start.X && finalX <= end.X && finalY >= start.Y && finalY <= end.Y)
                {
                    ret[finalX, finalY] = TerrainType.Route;
                }
                error -= dy;
                if (error < 0)
                {
                    y += ystep;
                    error += dx;
                }
            }
        }

        public void DrawWireframe(Point start, Point end, Transform tr, TerrainType[,] ret)
        {
            foreach (var p in Points)
            {
                var b = tr.ToBitmapCoords(p);
                ret[b.X, b.Y] = TerrainType.Route;
            }

            int last = Points.Count - 1;
            for (int i = 0; i < Points.Count; i++)
            {
                var pt1 = Points[last];
                var pt2 = Points[i];

                DrawLine(pt1, pt2, start, end, ret);
                last = i;
            }
        }
    }
}