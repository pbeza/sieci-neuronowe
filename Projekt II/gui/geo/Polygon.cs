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

        private List<PointD> Points { get; set; }

        private static double CrossProductLength(double Ax, double Ay, double Bx, double By, double Cx, double Cy)
        {
            double BAx = Ax - Bx;
            double BAy = Ay - By;
            double BCx = Cx - Bx;
            double BCy = Cy - By;

            return (BAx * BCy - BAy * BCx);
        }

        private static double DotProduct(double Ax, double Ay, double Bx, double By, double Cx, double Cy)
        {
            double BAx = Ax - Bx;
            double BAy = Ay - By;
            double BCx = Cx - Bx;
            double BCy = Cy - By;

            return (BAx * BCx + BAy * BCy);
        }

        private static double GetAngle(double Ax, double Ay, double Bx, double By, double Cx, double Cy)
        {
            double dot_product = DotProduct(Ax, Ay, Bx, By, Cx, Cy);
            double cross_product = CrossProductLength(Ax, Ay, Bx, By, Cx, Cy);
            return Math.Atan2(cross_product, dot_product);
        }

        public bool PointInPolygon(double X, double Y)
        {
            int max_point = Points.Count - 1;
            double total_angle = GetAngle(Points[max_point].X, Points[max_point].Y, X, Y, Points[0].X, Points[0].Y);

            for (int i = 0; i < max_point; i++)
            {
                total_angle += GetAngle(Points[i].X, Points[i].Y, X, Y, Points[i + 1].X, Points[i + 1].Y);
            }

            return (Math.Abs(total_angle) > 0.000001);
        }
    }
}