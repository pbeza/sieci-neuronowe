namespace gui.geo
{
    using OsmSharp.Math.Geo;
    using OsmSharp.Osm;

    public class Route : IBoundingBox
    {
        private readonly GeoCoordinateBox boundingBox;

        public Way way { get; private set; }

        public Route(Way w, GeoCoordinateBox box)
        {
            way = w;
            boundingBox = box;
        }

        public GeoCoordinateBox GetBounds()
        {
            return boundingBox;
        }
    }
}