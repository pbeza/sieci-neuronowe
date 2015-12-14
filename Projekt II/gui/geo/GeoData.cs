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

        private readonly Dictionary<long, Way> ways;

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

        private void CacheBuildings()
        {
            foreach (var keyValuePair in ways)
            {
                var way = keyValuePair.Value;
                if (way.Tags == null || !way.Tags.ContainsKey("building"))
                {
                    continue;
                }
                CacheBuilding(way);
            }

            foreach (var keyValuePair in relations)
            {
                var relation = keyValuePair.Value;
                if (relation.Tags == null || !relation.Tags.ContainsKey("building"))
                {
                    continue;
                }

                if (!relation.Id.HasValue)
                {
                    continue;
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
                        if (this.ways.TryGetValue(relationMember.MemberId.Value, out way))
                        {
                            this.CacheBuilding(way);
                        }
                    }
                }
            }

            boundsCheck = new BoundsCheck<Building>(new List<Building>(buildings.Values));
        }

        public TerrainType[,] GetTypesInArea(GeoCoordinateBox bounds, int resolutionX, int resolutionY)
        {
            var ret = new TerrainType[resolutionX, resolutionY];
            var resolution = new Point(resolutionX, resolutionY);
            var offset = new PointD(bounds.MinLat, bounds.MinLon);
            var step = new PointD(bounds.DeltaLat / resolutionX, bounds.DeltaLon / resolutionY);
            var inBounds = boundsCheck.GetValuesInBounds(bounds);
            var enumerable = inBounds as IList<Building> ?? inBounds.ToList();
            var boundsSize = enumerable.Count();
            for (int index = 0; index < enumerable.Count; index++)
            {
                var building = enumerable[index];
                var buildingBounds = building.GetBounds();
                var poly = this.WayToPolygon(building.way);
                var startGlobal = new PointD(buildingBounds.MinLat, buildingBounds.MinLon);
                var start = ToBitmapCoords(startGlobal, offset, step, resolution);
                var endGlobal = new PointD(buildingBounds.MaxLat, buildingBounds.MaxLon);
                var end = ToBitmapCoords(endGlobal, offset, step, resolution);
                if (start.X >= end.X - 1 && start.Y >= end.Y - 1)
                    //if(true)
                {
                    ret[start.X, start.Y] = TerrainType.Building;
                }
                else
                {
                    DrawPolygon(bounds, start, end, resolution, offset, step, poly, ret);
                }
            }
            return ret;
        }

        private static Point ToBitmapCoords(PointD p, PointD offset, PointD step, Point resolution)
        {
            var x = (int)((p.X - offset.X) / step.X);
            var y = (int)((p.Y - offset.Y) / step.Y);
            x = Math.Max(0, Math.Min(x, resolution.X - 1));
            y = Math.Max(0, Math.Min(y, resolution.Y - 1));
            return new Point(x, y);
        }

        private static PointD ToBitmapCoordsD(PointD p, PointD offset, PointD step, Point resolution)
        {
            var x = ((p.X - offset.X) / step.X);
            var y = ((p.Y - offset.Y) / step.Y);
            x = Math.Max(0, Math.Min(x, resolution.X - 1));
            y = Math.Max(0, Math.Min(y, resolution.Y - 1));
            return new PointD(x, y);
        }

        private static void DrawPolygon(
            GeoCoordinateBox bounds,
            Point start,
            Point end,
            Point resolution,
            PointD offset,
            PointD step,
            Polygon poly,
            TerrainType[,] ret)
        {
            foreach (var p in poly.Points)
            {
                var b = ToBitmapCoords(p, offset, step, resolution);
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
            int polyCorners = poly.Points.Count;
            var pts = poly.Points.Select(x => ToBitmapCoordsD(x, offset, step, resolution)).ToArray();
            for (int pixelY = start.Y; pixelY < end.Y; pixelY++)
            {
                //  Build a list of nodes.
                List<int> nodeX = new List<int>(pts.Length);
                int j = polyCorners - 1;
                for (int i = 0; i < polyCorners; i++)
                {
                    if (pts[i].Y < pixelY && pts[j].Y >= pixelY
                        || pts[j].Y < pixelY && pts[i].Y >= pixelY)
                    {
                        nodeX.Add((int)(pts[i].X + (pixelY - pts[i].Y) / (pts[j].Y - pts[i].Y) * (pts[j].X - pts[i].X)));
                    }
                    j = i;
                }

                nodeX.Sort((a, b) => (a - b));

                //  Fill the pixels between node pairs.
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
                }
            }
        }
    }
}