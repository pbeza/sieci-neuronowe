namespace gui.geo
{
    #region

    using System;
    using System.Collections.Generic;

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
    }
}