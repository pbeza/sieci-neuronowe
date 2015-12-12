namespace gui.geo
{
    #region

    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Windows;

    using OsmSharp.Math.Geo;
    using OsmSharp.Osm;
    using OsmSharp.Osm.Xml.Streams;

    #endregion

    public class GeoData
    {
        private readonly string path;

        private readonly Dictionary<long, Node> nodes;

        private readonly Dictionary<long, Relation> relations;

        private readonly Dictionary<long, Way> ways;

        private readonly Dictionary<long, Building> buildings;

        private BoundsCheck<Building> boundsCheck;

        public GeoData(string loadPath)
        {
            nodes = new Dictionary<long, Node>();
            relations = new Dictionary<long, Relation>();
            ways = new Dictionary<long, Way>();
            buildings = new Dictionary<long, Building>();
            path = loadPath;
            try
            {
                Load();
            }
            catch (Exception e)
            {
                MessageBox.Show("Error: " + e);
            }
        }

        private void Load()
        {
            var xmlStream = new XmlOsmStreamSource(new FileInfo(path).OpenRead());

            foreach (var geo in xmlStream)
            {
                if (geo.Id == null)
                {
                    continue;
                }
                var id = geo.Id.Value;
                switch (geo.Type)
                {
                    case OsmGeoType.Node:
                        nodes[id] = (Node)geo;
                        break;
                    case OsmGeoType.Way:
                        ways[id] = (Way)geo;
                        break;
                    case OsmGeoType.Relation:
                        relations[id] = (Relation)geo;
                        break;
                    default:
                        throw new Exception("Unexpected type in XML: " + geo.Type);
                }
            }

            CacheBuildings();
        }

        private GeoCoordinateBox GetBoundingBox(Way w)
        {
            return new GeoCoordinateBox(GetNodeCoords(w));
        }

        private List<GeoCoordinate> GetNodeCoords(Way way)
        {
            var ret = new List<GeoCoordinate>(way.Nodes.Count);
            foreach (var nodeId in way.Nodes)
            {
                Node node;
                if (!nodes.TryGetValue(nodeId, out node))
                {
                    continue;
                }

                ret.Add(node.Coordinate);
            }

            return ret;
        }

        private void CacheBuildings()
        {
            foreach (var keyValuePair in ways)
            {
                var way = keyValuePair.Value;
                if (way.Tags == null || !way.Tags.ContainsKey("building"))
                {
                    continue;
                }
                if (!way.Id.HasValue)
                {
                    continue;
                }
                var id = way.Id.Value;
                var box = GetBoundingBox(way);
                buildings.Add(id, new Building(keyValuePair.Value, box));
            }

            boundsCheck = new BoundsCheck<Building>(new List<Building>(buildings.Values));
        }

        public TerrainType[,] GetTypesInArea(GeoCoordinateBox bounds, int resolutionX, int resolutionY)
        {
            var ret = new TerrainType[resolutionX, resolutionY];
            var stepX = bounds.DeltaLat / resolutionX;
            var stepY = bounds.DeltaLon / resolutionY;
            var inBounds = boundsCheck.GetValuesInBounds(bounds);
            foreach (var building in inBounds)
            {
                var buildingBounds = building.GetBounds();
                var startX = (int)((buildingBounds.MinLat - bounds.MinLat) / stepX);
                var startY = (int)((buildingBounds.MinLon - bounds.MinLon) / stepY);
                startX = Math.Max(0, startX);
                startY = Math.Max(0, startY);
                var endX = (int)((buildingBounds.MaxLat - bounds.MinLat) / stepX) + 1;
                var endY = (int)((buildingBounds.MaxLon - bounds.MinLon) / stepY) + 1;
                endX = Math.Min(resolutionX, endX);
                endY = Math.Min(resolutionY, endY);
                for (int x = startX; x < endX; x++)
                {
                    for (int y = startY; y < endY; y++)
                    {
                        //if (poly.PointInPolygon(xLocal, yLocal))
                        {
                            ret[x, y] = TerrainType.Building;
                        }
                    }
                }
            }
            return ret;
        }
    }
}