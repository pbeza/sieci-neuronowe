using System.Linq;

namespace gui.geo
{
    #region

    using System;
    using System.Collections.Generic;

    using OsmSharp.Math.Geo;

    #endregion

    public class BoundsCheck<T>
        where T : IBoundingBox
    {
        private readonly List<T> unsorted;
        private readonly List<Tuple<double, int>> minLatSorted;
        private readonly List<Tuple<double, int>> maxLatSorted;

        public BoundsCheck(List<T> unsortedIn)
        {
            unsorted = unsortedIn;
            minLatSorted = new List<Tuple<double, int>>(unsorted.Count);
            maxLatSorted = new List<Tuple<double, int>>(unsorted.Count);
            for (int i = 0; i < unsorted.Count; i++)
            {
                var bnd = unsorted[i].GetBounds();
                minLatSorted.Add(Tuple.Create(bnd.MinLat, i));
                maxLatSorted.Add(Tuple.Create(bnd.MaxLat, i));
            }

            minLatSorted.Sort((a, b) => Math.Sign(a.Item1 - b.Item1));
            maxLatSorted.Sort((a, b) => Math.Sign(a.Item1 - b.Item1));
        }

        private int LowerBound(List<Tuple<double, int>> list, double bound)
        {
            int first = 0;
            int last = minLatSorted.Count;
            int count = last - first;
            while (count > 0)
            {
                int it = first;
                int step = count / 2;
                it += step;
                if (list[it].Item1 < bound)
                {
                    first = ++it;
                    count -= step + 1;
                }
                else
                {
                    count = step;
                }
            }
            return first;
        }

        private int UpperBound(List<Tuple<double, int>> list, double bound)
        {
            int first = 0;
            int last = minLatSorted.Count;
            int count = last - first;
            while (count > 0)
            {
                int it = first;
                int step = count / 2;
                it += step;
                if (list[it].Item1 < bound)
                {
                    first = ++it;
                    count -= step + 1;
                }
                else
                {
                    count = step;
                }
            }
            return first + 1;
        }

        public IEnumerable<T> GetValuesInBounds(GeoCoordinateBox box)
        {
            int firstMin = LowerBound(minLatSorted, box.MinLat);
            int lastMin = UpperBound(minLatSorted, box.MaxLat);
            lastMin = Math.Min(minLatSorted.Count - 1, lastMin);
            var ids = new HashSet<int>();
            for (int i = firstMin; i <= lastMin; i++)
            {
                ids.Add(minLatSorted[i].Item2);
            }

            int firstMax = LowerBound(maxLatSorted, box.MinLat);
            int lastMax = UpperBound(maxLatSorted, box.MaxLat);
            lastMax = Math.Min(minLatSorted.Count - 1, lastMax);
            for (int i = firstMax; i <= lastMax; i++)
            {
                ids.Add(minLatSorted[i].Item2);
            }

            return ids.Select(id => this.unsorted[id]);
        }
    }
}