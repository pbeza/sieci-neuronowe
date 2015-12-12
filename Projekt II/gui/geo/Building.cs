namespace gui.geo
{
    #region

    using OsmSharp.Math.Geo;
    using OsmSharp.Osm;

    #endregion

    public class Building : IBoundingBox
    {
        private GeoCoordinateBox boundingBox;

        public Way way { get; private set; }

        public Building(Way w, GeoCoordinateBox box)
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