namespace gui.geo
{
    #region

    using System.Windows.Media;
    using System.Windows.Media.Imaging;

    #endregion

    public static class BitmapHelper
    {
        private const int DpiX = 96, DpiY = 96;

        public static BitmapSource FromTypeArray(TerrainType[,] arr)
        {
            var resolutionX = arr.GetLength(0);
            var resolutionY = arr.GetLength(1);
            var byteArray = new byte[arr.Length * 3];
            for (var y = 0; y < resolutionY; y++)
            {
                for (var x = 0; x < resolutionX; x++)
                {
                    if (arr[x, y] == TerrainType.Nothing)
                    {
                        continue;
                    }

                    var index = (y * resolutionX + x) * 3;
                    byteArray[index] = 0;
                    byteArray[index + 1] = (byte)(arr[x, y] == TerrainType.Building ? 255 : 0);
                    byteArray[index + 2] = (byte)(arr[x, y] == TerrainType.Route ? 255 : 0);
                }
            }

            return BitmapSource.Create(
                resolutionX,
                resolutionY,
                DpiX,
                DpiY,
                PixelFormats.Rgb24, 
                null,
                byteArray,
                resolutionX * 3);
        }
    }
}