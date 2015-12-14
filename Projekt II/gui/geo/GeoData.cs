namespace gui.geo
{
    #region

    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Windows;

    using OsmSharp.Math.Geo;
    using OsmSharp.Osm;
    using OsmSharp.Osm.Xml.Streams;

    using Point = System.Drawing.Point;

    #endregion

    public class GeoData
    {
        private readonly Dictionary<long, Building> buildings;

        private readonly Dictionary<long, Node> nodes;

        private readonly string path;

        private readonly Dictionary<long, Relation> relations;

        private readonly Dictionary<long, Route> routes;

        private readonly Dictionary<long, Way> ways;

        private BoundsCheck<Building> buildingBounds;

        private BoundsCheck<Route> routeBounds;

        public GeoData(string loadPath)
        {
            nodes = new Dictionary<long, Node>();
            relations = new Dictionary<long, Relation>();
            ways = new Dictionary<long, Way>();
            buildings = new Dictionary<long, Building>();
            routes = new Dictionary<long, Route>();
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

            this.CacheFeatures();
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

        private Polygon WayToPolygon(Way w)
        {
            List<PointD> poly = new List<PointD>(w.Nodes.Count);
            poly.AddRange(
                w.Nodes.Select(nodeId => this.nodes[nodeId])
                    .Select(node => new PointD((double)node.Latitude, (double)node.Longitude)));

            return new Polygon(poly);
        }

        private void CacheBuilding(Way way)
        {
            if (!way.Id.HasValue || buildings.ContainsKey(way.Id.Value))
            {
                return;
            }
            var id = way.Id.Value;
            var box = GetBoundingBox(way);
            buildings.Add(id, new Building(way, box));
        }

        private void CacheRoute(Way way)
        {
            if (!way.Id.HasValue || this.routes.ContainsKey(way.Id.Value))
            {
                return;
            }
            var id = way.Id.Value;
            var box = GetBoundingBox(way);
            this.routes.Add(id, new Route(way, box));
        }

        private void CacheFeatures()
        {
            foreach (var keyValuePair in ways)
            {
                var way = keyValuePair.Value;
                if (way.Tags == null || !way.Tags.ContainsKey("building"))
                {
                    continue;
                }
                if (way.Tags.ContainsKey("building"))
                {
                    CacheBuilding(way);
                }
                else if (way.Tags.ContainsKey("route"))
                {
                    this.CacheRoute(way);
                }
            }

            foreach (var keyValuePair in relations)
            {
                var relation = keyValuePair.Value;
                if (relation.Tags == null)
                {
                    continue;
                }

                if (!relation.Id.HasValue)
                {
                    continue;
                }

                bool isBuilding = relation.Tags.ContainsKey("building");
                if (!isBuilding)
                {
                    if (!relation.Tags.ContainsKey("route"))
                    {
                        continue;
                    }
                }

                foreach (var value in relations.Values)
                {
                    foreach (var relationMember in value.Members)
                    {
                        if (relationMember.MemberType != OsmGeoType.Way || !relationMember.MemberId.HasValue)
                        {
                            continue;
                        }
                        Way way;
                        if (!this.ways.TryGetValue(relationMember.MemberId.Value, out way))
                        {
                            continue;
                        }

                        if (isBuilding)
                        {
                            this.CacheBuilding(way);
                        }
                        else
                        {
                            this.CacheRoute(way);
                        }
                    }
                }
            }

            buildingBounds = new BoundsCheck<Building>(new List<Building>(buildings.Values));
            routeBounds = new BoundsCheck<Route>(new List<Route>(routes.Values));
        }

        public TerrainType[,] GetTypesInArea(GeoCoordinateBox bounds, int resolutionX, int resolutionY)
        {
            var ret = new TerrainType[resolutionX, resolutionY];
            var resolution = new Point(resolutionX, resolutionY);
            var offset = new PointD(bounds.MinLat, bounds.MinLon);
            var step = new PointD(bounds.DeltaLat / resolutionX, bounds.DeltaLon / resolutionY);
            var inBoundsBuildings = buildingBounds.GetValuesInBounds(bounds);
            Point start;
            Point end;
            foreach (var bld in inBoundsBuildings)
            {
                var tr = new Transform(resolution, offset, step);
                tr.TranslateBounds(bld, out start, out end);
                if (start.X >= end.X - 1 && start.Y >= end.Y - 1)
                {
                    ret[start.X, start.Y] = TerrainType.Building;
                }
                else
                {
                    var poly = WayToPolygon(bld.way);
                    poly.Draw(start, end, tr, ret);
                }
            }

            var inBoundsRoutes = routeBounds.GetValuesInBounds(bounds);
            foreach (var route in inBoundsRoutes)
            {
                var tr = new Transform(resolution, offset, step);
                tr.TranslateBounds(route, out start, out end);
                if (start.X >= end.X - 1 && start.Y >= end.Y - 1)
                {
                    ret[start.X, start.Y] = TerrainType.Building;
                }
                else
                {
                    var poly = WayToPolygon(route.way);
                    poly.DrawWireframe(start, end, tr, ret);
                }
            }
            return ret;
        }
    }
}