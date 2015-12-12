namespace gui.geo
{
    using OsmSharp.Math.Geo;

    public interface IBoundingBox
    {
        GeoCoordinateBox GetBounds();
    }
}