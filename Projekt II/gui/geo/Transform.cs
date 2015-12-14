namespace gui.geo
{
    #region

    using System;
    using System.Drawing;

    #endregion

    public class Transform
    {
        private PointD offset;

        private Point resolution;

        private PointD step;

        public Transform(Point res, PointD ofst, PointD stp)
        {
            resolution = res;
            offset = ofst;
            step = stp;
        }

        public void TranslateBounds(
            IBoundingBox bld,
            out Point start,
            out Point end)
        {
            var bldBounds = bld.GetBounds();
            var startGlobal = new PointD(bldBounds.MinLat, bldBounds.MinLon);
            start = ToBitmapCoords(startGlobal);
            var endGlobal = new PointD(bldBounds.MaxLat, bldBounds.MaxLon);
            end = ToBitmapCoords(endGlobal);
        }

        public Point ToBitmapCoords(PointD p)
        {
            var x = (int)((p.X - offset.X) / step.X);
            var y = (int)((p.Y - offset.Y) / step.Y);
            x = Math.Max(0, Math.Min(x, resolution.X - 1));
            y = Math.Max(0, Math.Min(y, resolution.Y - 1));
            return new Point(x, y);
        }

        public PointD ToBitmapCoordsD(PointD p)
        {
            var x = ((p.X - offset.X) / step.X);
            var y = ((p.Y - offset.Y) / step.Y);
            x = Math.Max(0, Math.Min(x, resolution.X - 1));
            y = Math.Max(0, Math.Min(y, resolution.Y - 1));
            return new PointD(x, y);
        }
    }
}