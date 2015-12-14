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
        private readonly List<Tuple<double, T>> minLatSorted;

        public BoundsCheck(IReadOnlyCollection<T> unsorted)
        {
            minLatSorted = new List<Tuple<double, T>>(unsorted.Count);
            foreach (var tp in unsorted.Select(elem => Tuple.Create(elem.GetBounds().MinLat, elem)))
            {
                minLatSorted.Add(tp);
            }

            minLatSorted.Sort((a, b) => Math.Sign(a.Item1 - b.Item1));
        }

        private int LowerBound(double bound)
        {
            int first = 0;
            int last = minLatSorted.Count;
            int count = last - first;
            while (count > 0)
            {
                int it = first;
                int step = count / 2;
                it += step;
                if (minLatSorted[it].Item1 < bound)
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

        private int UpperBound(double bound)
        {
            int first = 0;
            int last = minLatSorted.Count;
            int count = last - first;
            while (count > 0)
            {
                int it = first;
                int step = count / 2;
                it += step;
                if (minLatSorted[it].Item1 < bound)
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
            int first = LowerBound(box.MinLat);
            int last = UpperBound(box.MaxLat);
            last = Math.Min(minLatSorted.Count - 1, last);
            for (int i = first; i <= last; i++)
            {
                var elem = minLatSorted[i].Item2;
                var bounds = elem.GetBounds();
                if (box.Overlaps(bounds))
                {
                    yield return elem;
                }
            }
        }
    }
}